"""pascal_voc dataset."""

import os
import xml.etree.ElementTree
import tensorflow as tf
import tensorflow_datasets as tfds

_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

_CITATION = """
"""

_VOC_LABELS = {
    'aeroplane': 0,
    'bicycle': 1,
    'bird': 2,
    'boat': 3,
    'bottle': 4,
    'bus': 5,
    'car': 6,
    'cat': 7,
    'chair': 8,
    'cow': 9,
    'diningtable': 10,
    'dog': 11,
    'horse': 12,
    'motorbike': 13,
    'person': 14,
    'pottedplant': 15,
    'sheep': 16,
    'sofa': 17,
    'train': 18,
    'tvmonitor': 19
}


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

            xmax = float(bndbox.find("xmax").text) - 1
            xmin = float(bndbox.find("xmin").text) - 1
            ymax = float(bndbox.find("ymax").text) - 1
            ymin = float(bndbox.find("ymin").text) - 1
            yield {
                "id":
                    _VOC_LABELS[label],
                "label":
                    label,
                "bbox":
                    tfds.features.BBox(xmin / width,
                                       ymin / height,
                                       xmax / width,
                                       ymax / height),
            }


class PascalVOC(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for pascal_voc dataset."""

    VERSION = tfds.core.Version('1.0.1')
    RELEASE_NOTES = {
        '1.0.1': 'Initial release.',
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
                        "label": tfds.features.ClassLabel(names=_VOC_LABELS.keys()),
                        "bbox": tfds.features.BBoxFeature()
                    }),
            }),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        paths = "D:\\"

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs=dict(data_path=paths, set_name="test")),
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs=dict(data_path=paths, set_name="trainval")),
        ]

    def _generate_examples(self, data_path, set_name):
        """Yields examples."""
        set_filepath = os.path.join(
            data_path,
            os.path.normpath("VOC2012+2007/ImageSets/Main/{}.txt".format(set_name)))

        with tf.io.gfile.GFile(set_filepath, "r") as f:
            for line in f:
                image_id = line.strip()
                example = self._generate_example(data_path, image_id)
                yield image_id, example

    @staticmethod
    def _generate_example(data_path, image_id):
        image_filepath = os.path.join(
            data_path,
            os.path.normpath("VOC2012+2007/JPEGImages/{}.jpg".format(image_id)))

        annon_filepath = os.path.join(
            data_path,
            os.path.normpath("VOC2012+2007/Annotations/{}.xml".format(image_id)))

        objects = list(_get_example_objects(annon_filepath))

        return {
            "image": image_filepath,
            "image/filename": image_id + ".jpg",
            "objects": objects,
        }
