# Building Tensorflow Datasets (tfds)

_建立自訂 tensorflow-dataset 提高訓練過程資料傳遞效率及解決多卡訓練問題。(適用 Tensorflow 2.6.0)_

## 目錄

1. [初始化](#1-初始化)
2. [建立](#2-建立)
3. [其他](#3-其他)

## 1. 初始化

以下由 Deep PCB 資料集作為範例(此資料集已預先轉換為 VOC 存放格式，
[DOWNLOAD](https://drive.google.com/file/d/1FxmlSW0A2QfYYfwMZer7aXOMBd_m6O7j/view?usp=sharing))
，透過下面指令的初始化，會自動生成資料夾及檔案。
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

:point_right: 需要 import 的有 :

```python
import os
import xml.etree.ElementTree
import tensorflow as tf
import tensorflow_datasets as tfds
```

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
                "label":
                    label,
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
                        "label": tfds.features.ClassLabel(names=_DPCB_LABELS.keys()),
                        "bbox": tfds.features.BBoxFeature()
                    }),
            }),
            citation=_CITATION,
        )
```

:point_right: 比較重要的是在切割資料集種類(訓練集 或 測試集)，我們是離線的轉換僅需改路徑即可。

```python
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
```

:point_right: 確認好資料集標註檔的解析及輸出後，就能開始建立。 --data_dir 為輸出路徑。

```
tfds build dpcb_db.py --data_dir ./
```

```
INFO[build.py]: Loading dataset dpcb_db.py from path: D:\build-tfds\dpcb_db\dpcb_db.py
INFO[dataset_info.py]: Load dataset info from .\dpcb_db\1.0.0
INFO[build.py]: download_and_prepare for dataset dpcb_db/1.0.0...
INFO[dataset_builder.py]: Reusing dataset dpcb_db (.\dpcb_db\1.0.0)
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
    dataset_size=78.20 MiB,
    features=FeaturesDict({
        'image': Image(shape=(None, None, 3), dtype=tf.uint8),
        'image/filename': Text(shape=(), dtype=tf.string),
        'objects': Sequence({
            'bbox': BBoxFeature(shape=(4,), dtype=tf.float32),
            'id': tf.int64,
            'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=6),
        }),
    }),
    supervised_keys=None,
    disable_shuffling=False,
    splits={
        'test': <SplitInfo num_examples=499, num_shards=1>,
        'train': <SplitInfo num_examples=1000, num_shards=1>,
    },
    citation="""""",
)
```

## 3. 讀取

### tfds.load

注意 data_dir 位置選擇，若沒有設置會導向 C:\使用者\tensorflow_datasets\。

```python
import tensorflow_datasets as tfds

train, test = tfds.load(name="dpcb_db", split=["train", "test"], data_dir="D:\\tensorflow_datasets")
```

這裡未來會放上問題及修正。