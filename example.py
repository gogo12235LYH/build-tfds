import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

_image_size = [512, 640, 768, 896, 1024, 1280, 1408]
_STRIDES = [8, 16, 32, 64, 128]
_ALPHA = 0.0


@tf.function
def _normalization_image(image, mode):
    if mode == 'ResNetV1':
        # Caffe
        image = image[..., ::-1]  # RGB -> BGR
        image -= [103.939, 116.779, 123.68]

    elif mode == 'ResNetV2':
        image /= 127.5
        image -= 1.

    elif mode == 'EffNet':
        image = image

    elif mode in ['DenseNet', 'SEResNet']:
        # Torch
        image /= 255.
        image -= [0.485, 0.456, 0.406]
        image /= [0.229, 0.224, 0.225]

    return image


def _fmap_shapes(phi: int = 0, level: int = 5):
    _img_size = int(phi * 128) + 512
    _strides = [int(2 ** (x + 3)) for x in range(level)]

    shapes = []

    for i in range(level):
        fmap_shape = _img_size // _strides[i]
        shapes.append([fmap_shape, fmap_shape])

    return shapes


@tf.function
def _image_transform(image, target_size=512, padding_value=.0):
    image_height, image_width = tf.shape(image)[0], tf.shape(image)[1]

    if image_height > image_width:
        scale = tf.cast((target_size / image_height), dtype=tf.float32)
        resized_height = target_size
        resized_width = tf.cast((tf.cast(image_width, dtype=tf.float32) * scale), dtype=tf.int32)
    else:
        scale = tf.cast((target_size / image_width), dtype=tf.float32)
        resized_height = tf.cast((tf.cast(image_height, dtype=tf.float32) * scale), dtype=tf.int32)
        resized_width = target_size

    image = tf.image.resize(
        image,
        (resized_height, resized_width),
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    offset_h = (target_size - resized_height) // 2
    offset_w = (target_size - resized_width) // 2

    # (h, w, c)
    pad = tf.stack(
        [
            tf.stack([offset_h, target_size - resized_height - offset_h], axis=0),
            tf.stack([offset_w, target_size - resized_width - offset_w], axis=0),
            tf.constant([0, 0]),
        ],
        axis=0
    )

    image = tf.pad(image, pad, constant_values=padding_value)

    return image, scale, [offset_h, offset_w]


@tf.function
def _bboxes_transform(bboxes, classes, scale, offset_hw, max_bboxes=20, padding=False):
    bboxes *= scale
    bboxes = tf.stack(
        [
            (bboxes[:, 0] + tf.cast(offset_hw[1], dtype=tf.float32)),
            (bboxes[:, 1] + tf.cast(offset_hw[0], dtype=tf.float32)),
            (bboxes[:, 2] + tf.cast(offset_hw[1], dtype=tf.float32)),
            (bboxes[:, 3] + tf.cast(offset_hw[0], dtype=tf.float32)),
            classes
        ],
        axis=-1,
    )

    if padding:
        # true_label_count
        bboxes_count = tf.shape(bboxes)[0]
        max_bbox_pad = tf.stack(
            [
                tf.stack([tf.constant(0), max_bboxes - bboxes_count], axis=0),
                tf.constant([0, 0]),
            ],
            axis=0
        )
        bboxes = tf.pad(bboxes, max_bbox_pad, constant_values=0.)

    else:
        bboxes_count = tf.shape(bboxes)[0]

    return bboxes, bboxes_count


@tf.function
def _clip_transformed_bboxes(image, bboxes, debug=False):
    image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)

    if debug:
        bboxes = tf.stack(
            [
                tf.clip_by_value(bboxes[:, 0] / image_shape[1], 0., 1.),  # x1
                tf.clip_by_value(bboxes[:, 1] / image_shape[0], 0., 1.),  # y1
                tf.clip_by_value(bboxes[:, 2] / image_shape[1], 0., 1.),  # x2
                tf.clip_by_value(bboxes[:, 3] / image_shape[0], 0., 1.),  # y2
                bboxes[:, -1]
            ],
            axis=-1
        )

    else:
        bboxes = tf.stack(
            [
                tf.clip_by_value(bboxes[:, 0], 0., image_shape[1] - 2),  # x1
                tf.clip_by_value(bboxes[:, 1], 0., image_shape[0] - 2),  # y1
                tf.clip_by_value(bboxes[:, 2], 1., image_shape[1] - 1),  # x2
                tf.clip_by_value(bboxes[:, 3], 1., image_shape[0] - 1),  # y2
                bboxes[:, -1]
            ],
            axis=-1
        )
    return bboxes


@tf.function
def compute_inputs(sample):
    image = tf.cast(sample["image"], dtype=tf.float32)
    image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    bboxes = tf.cast(sample["objects"]["bbox"], dtype=tf.float32)
    classes = tf.cast(sample["objects"]["label"], dtype=tf.float32)

    bboxes = tf.stack(
        [
            bboxes[:, 0] * image_shape[1],
            bboxes[:, 1] * image_shape[0],
            bboxes[:, 2] * image_shape[1],
            bboxes[:, 3] * image_shape[0],
        ],
        axis=-1
    )
    return image, image_shape, bboxes, classes


def preprocess_data(
        phi: int = 0,
        mode: str = "ResNetV1",
        padding_value: float = 128.,
        debug: bool = False,
):
    """Applies preprocessing step to a single sample

    ref: https://keras.io/examples/vision/retinanet/#preprocessing-data

    """

    def _preprocess_data(sample):
        #
        image, image_shape, bboxes, classes = compute_inputs(sample)

        # You can put some augmentation method here.

        # Transforming image and bboxes into fixed-size.
        image, scale, offset_hw = _image_transform(image, _image_size[phi], padding_value)
        image = _normalization_image(image, mode) if not debug else image

        # Clipping bboxes
        bboxes, bboxes_count = _bboxes_transform(bboxes, classes, scale, offset_hw, padding=False)
        bboxes = _clip_transformed_bboxes(image, bboxes, debug=debug)

        return image, bboxes

    return _preprocess_data


def inputs_targets(image, bboxes):
    inputs = {
        "image": image,
        "bboxes": bboxes,
    }
    return inputs


def create_pipeline(phi=0, mode="ResNetV1", db="DPCB", batch_size=1, debug=False):
    autotune = tf.data.AUTOTUNE
    _buffer = 1000

    if db == "DPCB":
        (train, test), ds_info = tfds.load(name="dpcb_db", split=["train", "test"], data_dir="D:/datasets/",
                                           with_info=True)
    elif db == "VOC":
        (train, test), ds_info = tfds.load(name="pascal_voc", split=["train", "test"], data_dir="D:/datasets/",
                                           with_info=True,
                                           shuffle_files=True)
    elif db == "VOC_mini":
        (train, test), ds_info = tfds.load(name="pascal_voc_mini", split=["train", "test"], data_dir="D:/datasets/",
                                           with_info=True,
                                           shuffle_files=True)
    else:
        train, test, ds_info = None, None, None

    train_examples = ds_info.splits["train"].num_examples
    test_examples = ds_info.splits["test"].num_examples
    print(f"[INFO] {db}: train( {train_examples} ), test( {test_examples} )")

    train = train.map(preprocess_data(phi=phi, mode=mode, debug=debug), num_parallel_calls=autotune)
    train = train.shuffle(_buffer, reshuffle_each_iteration=True)
    train = train.repeat()
    train = train.padded_batch(batch_size=batch_size, padding_values=(0.0, 0.0), drop_remainder=True)
    train = train.map(inputs_targets, num_parallel_calls=autotune)
    train = train.prefetch(autotune)

    return train, test


if __name__ == '__main__':
    eps = 10
    bs = 4

    train_t, test_t = create_pipeline(
        phi=1,
        batch_size=bs,
        debug=True,
        db="VOC"
    )

    """ """
    # for ep in range(eps):
    #     for step, inputs_batch in enumerate(train_t):
    #         # _cls = inputs_batch['cls_target'].numpy()
    #         # _loc = inputs_batch['loc_target'].numpy()
    #         # _ind = inputs_batch['ind_target'].numpy()
    #         _int = inputs_batch['bboxes_cnt'].numpy()
    #
    #         print(f"Ep: {ep + 1}/{eps} - {step + 1}, Batch: {_int.shape[0]}, {_int[:, 0]}")
    #
    #         if np.min(_int) == 0:
    #             break
    #
    #         # if step > (16551 // bs) - 3:
    #         #     min_cnt = np.min(_int)
    #         #     print(f"Ep: {ep + 1}/{eps} - {step + 1}, Batch: {_int.shape[0]}, {min_cnt}")

    """ """

    # iterations = 1
    # for step, inputs_batch in enumerate(train_t):
    #     if (step + 1) > iterations:
    #         break
    #
    #     print(f"[INFO] {step + 1} / {iterations}")
    #
    #     _cls = inputs_batch['cls_target'].numpy()
    #     _loc = inputs_batch['loc_target'].numpy()
    #     _ind = inputs_batch['ind_target'].numpy()
    #     _int = inputs_batch['bboxes_cnt'].numpy()
    #
    # p7_cls = np.reshape(_cls[0, 8500:, -2], (5, 5))
    # p6_cls = np.reshape(_cls[0, 8400:8500, -2], (10, 10))
    # p5_cls = np.reshape(_cls[0, 8000:8400, -2], (20, 20))
    #
    # p7_loc = np.reshape(_loc[0, 8500:, -2], (5, 5))
    # p6_loc = np.reshape(_loc[0, 8400:8500, -2], (10, 10))
    # p5_loc = np.reshape(_loc[0, 8000:8400, -2], (20, 20))

    """ """

    import matplotlib.pyplot as plt

    iterations = 10
    print('test')
    plt.figure(figsize=(10, 8))
    for step, inputs_batch in enumerate(train_t):
        if (step + 1) > iterations:
            break

        print(f"[INFO] {step + 1} / {iterations}")

        _images = inputs_batch['image'].numpy()
        _bboxes = inputs_batch['bboxes'].numpy()

        _bboxes = tf.stack(
            [
                _bboxes[..., 1],
                _bboxes[..., 0],
                _bboxes[..., 3],
                _bboxes[..., 2],
            ],
            axis=-1
        )

        colors = np.array([[255.0, 0.0, 0.0]])
        _images = tf.image.draw_bounding_boxes(
            _images,
            _bboxes,
            colors=colors
        )

        for i in range(bs):
            plt.subplot(2, 2, i + 1)
            plt.imshow(_images[i].numpy().astype("uint8"))
            # print(bboxes[i])
        plt.tight_layout()
        plt.pause(1)
        # plt.close()

    """ """

    # tfds.benchmark(train_t, batch_size=bs)
    # tfds.benchmark(train_t, batch_size=bs)

    # image : (Batch, None, None, 3)
    # bboxes : (Batch, None, 5)
    # bboxes_count : (Batch, 1)
    # fmaps_shape : (Batch, 5, 2)
