# Building Tensorflow Datasets (tfds)

_建立自訂 tensorflow-dataset 提高訓練過程資料傳遞效率及解決多卡訓練問題。(適用 Tensorflow 2.6.0)_

## 目錄

1. [介紹](#1-介紹)
2. [建立](#2-建立)
3. [其他](#3-其他)

## 1. 介紹

在此章節裡將介紹使用 tfds 進行初始化及建立自己的資料集型態。

以下由 Deep PCB 資料集作為範例，透過下面指令的初始化，會自動生成資料夾及檔案。
[可參考這裡](https://www.tensorflow.org/datasets/cli#tfds_new_implementing_a_new_dataset)

```
tfds new dpcb_db
```

如果只要將舊有檔案進行轉換，僅需 dpcb_db.py 即可。

```commandline
 D:\database\dpcb_db 的目錄

2021/09/27  下午 10:29    <DIR>          .
2021/09/27  下午 10:29    <DIR>          ..
2021/09/27  下午 09:15               157 checksums.tsv
2021/09/27  下午 10:13             4,521 dpcb_db.py
2021/09/27  下午 09:37               750 dpcb_db_test.py
2021/09/27  下午 09:15                55 __init__.py
2021/09/27  下午 09:41    <DIR>          __pycache__
               4 個檔案           5,483 位元組
               3 個目錄  122,679,873,536 位元組可用
```

## 2. 建立

本範例僅需修改 dpcb_db.py，在舊有形式上採用 PASCAL VOC 的編排方式進行訓練，老大提供原始碼我們就參考一下， [這裡](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/object_detection/voc.py) 。

:point_right: 所有標記檔案都為 .xml 形式，以下是解析關鍵字及抽取資料集輸出形式。

```python
def _get_example_objects(annon_filepath):
    """Function to get all the objects from the annotation XML file."""
    with tf.io.gfile.GFile(annon_filepath, "r") as f:
        root = xml.etree.ElementTree.parse(f).getroot()

        # Disable pytype to avoid attribute-error due to find returning
        # Optional[Element]
        # pytype: disable=attribute-error
        size = root.find("size")
        width = float(size.find("width").text)
        height = float(size.find("height").text)

        for obj in root.findall("object"):
            # Get object's label name.
            label = obj.find("name").text.lower()

            bndbox = obj.find("bndbox")

            xmax = float(bndbox.find("xmax").text)
            xmin = float(bndbox.find("xmin").text)
            ymax = float(bndbox.find("ymax").text)
            ymin = float(bndbox.find("ymin").text)
            yield {
                "id":
                    _DPCB_LABELS[label],
                "bbox":
                    tfds.features.BBox(ymin / height,
                                       xmin / width,
                                       ymax / height,
                                       xmax / width),
            }
```

:point_right: 整體資料集形式如下，其中 objects 對應上面每張影像上的所有目標。

```python
class DpcbDb(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for dpcb_db dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                "image":
                    tfds.features.Image(),
                "image/filename":
                    tfds.features.Text(),
                "objects":
                    tfds.features.Sequence({
                        "id": tf.int64,
                        "bbox": tfds.features.BBoxFeature()
                    }),
            }),
            citation=_CITATION,
        )
```


確認好資料集標註檔的解析及輸出後，就能進行建立。

```
tfds build dpcb_db.py --data_dir ./
```

```commandline
INFO[build.py]: Loading dataset dpcb_db.py from path: D:\database\dpcb_db\dpcb_db.py
INFO[build.py]: download_and_prepare for dataset dpcb_db/1.0.0...
INFO[dataset_builder.py]: Generating dataset dpcb_db (.\dpcb_db\1.0.0)
Downloading and preparing dataset Unknown size (download: Unknown size, generated: Unknown size, total: Unknown size) to .\dpcb_db\1.0.0...
INFO[tfrecords_writer.py]: Done writing dpcb_db-test.tfrecord. Number of examples: 499 (shards: [499])
INFO[tfrecords_writer.py]: Done writing dpcb_db-train.tfrecord. Number of examples: 1000 (shards: [1000])
Dataset dpcb_db downloaded and prepared to .\dpcb_db\1.0.0. Subsequent calls will reuse this data.
INFO[build.py]: Dataset generation complete...

tfds.core.DatasetInfo(
    name='dpcb_db',
    full_name='dpcb_db/1.0.0',
    description="""
    Description is **formatted** as markdown.

    It should also contain any processing which has been applied (if any),
    (e.g. corrupted example skipped, images cropped,...):
    """,
    homepage='https://www.tensorflow.org/datasets/catalog/dpcb_db',
    data_path='.\\dpcb_db\\1.0.0',
    download_size=Unknown size,
    dataset_size=78.16 MiB,
    features=FeaturesDict({
        'image': Image(shape=(None, None, 3), dtype=tf.uint8),
        'image/filename': Text(shape=(), dtype=tf.string),
        'objects': Sequence({
            'bbox': BBoxFeature(shape=(4,), dtype=tf.float32),
            'id': tf.int64,
        }),
    }),
    supervised_keys=('image', 'label'),
    disable_shuffling=False,
    splits={
        'test': <SplitInfo num_examples=499, num_shards=1>,
        'train': <SplitInfo num_examples=1000, num_shards=1>,
    },
    citation="""""",
)

```

## 3. 其他

這裡未來會放上問題及修正。