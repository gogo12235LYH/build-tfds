# Tensorflow Datasets (tfds)

_建立自訂 tensorflow-dataset 提高訓練過程資料傳遞效率及解決多卡訓練問題。(適用 Tensorflow 2.6.0)_


---

## Updates

* 2022-04-28 -> 新增 VOC_mini
* 2022-04-09 -> 新增 VOC 轉 tfrecord.

---

## 目錄

1. [初始化](#1-初始化)
2. [建立](#2-建立)
3. [下載](#3-下載)
4. [Tips](#4-Tips)

## 1. 初始化

以下由 Deep PCB 資料集作為範例(此資料集已預先轉換為 VOC 存放格式，
[DOWNLOAD](https://drive.google.com/file/d/1FxmlSW0A2QfYYfwMZer7aXOMBd_m6O7j/view?usp=sharing))
，透過下面指令的初始化，會自動生成資料夾及檔案。
[可參考這裡](https://www.tensorflow.org/datasets/cli#tfds_new_implementing_a_new_dataset)

### Deep PCB 資料集 (VOC Like) >>>> [Download](https://drive.google.com/file/d/12MTL3seeA4ZqnVzw1oWwdmE5KGxyAcqC/view?usp=sharing)

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

本範例僅需修改 dpcb_db.py，在舊有形式上採用 PASCAL VOC 的格式進行訓練，可參考官網
-> [這裡](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/object_detection/voc.py) 。

:point_right: 所有標記檔案都為 .xml 形式，以下是解析關鍵字及抽取資料集輸出形式。 注意，bbox 座標要求為 0~1，這裡直接將座標 x 與 y 分別除以 影像的寬與長。

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

            xmax = float(bndbox.find("xmax").text) - 1
            xmin = float(bndbox.find("xmin").text) - 1
            ymax = float(bndbox.find("ymax").text) - 1
            ymin = float(bndbox.find("ymin").text) - 1
            yield {
                "id":
                    _DPCB_LABELS[label],
                "label":
                    label,
                "bbox":
                    tfds.features.BBox(xmin / width,
                                       ymin / height,
                                       xmax / width,
                                       ymax / height),
            }
```

:point_right: 整體資料集形式如下，其中 objects 對應上面每張影像上的所有目標。

```python
class DpcbDb(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for dpcb_db dataset."""

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

paths = "D:\\"  # < -- 修改路徑

return [
    tfds.core.SplitGenerator(
        name=tfds.Split.TEST,
        gen_kwargs=dict(data_path=paths, set_name="test")),
    tfds.core.SplitGenerator(
        name=tfds.Split.TRAIN,
        gen_kwargs=dict(data_path=paths, set_name="trainval")),
]
```

:point_right: 確認好資料集標註檔的解析及輸出後，就能開始建立。 --data_dir 為輸出路徑
(預設為 C:\user\tensorflow_datasets\)。

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

## 3. 下載

| Dataset              | Description         | Download          |
|----------------------|---------------------|-------------------|
| Deep PCB             | 二值化之印刷電路瑕疵資料集，共六種瑕疵 | [DPCB](https://drive.google.com/file/d/12MTL3seeA4ZqnVzw1oWwdmE5KGxyAcqC/view?usp=sharing)          |
| Deep PCB (tfrecord)  | -                   | [DPCB_tfds](https://drive.google.com/file/d/15y3md6zUVFX6cmI_pXYeBF33JsYpje00/view?usp=sharing)     |
| VOC(2007+2012)       | 彩色泛用大型資料集，共20種分類    | [VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)           |
| VOC(07+12, tfrecord) | -                   | [VOC_tfds](https://drive.google.com/file/d/1nF6OO2NZTPMwp-Sf1QTMcTdgHmPXX0b9/view?usp=sharing)      |
| VOC_mini(tfrecord)   | -                   | [VOC_mini_tfds](https://drive.google.com/file/d/1wXBX0p50jlJKpuOL4Vwo3g9xcFg0nT1X/view?usp=sharing) |

* original Deep PCB dataset -> [https://github.com/tangsanli5201/DeepPCB](https://github.com/tangsanli5201/DeepPCB)

### Deep PCB Example
![image](https://github.com/gogo12235LYH/build-tfds/blob/master/images/dpcb.png)

### PASCAL VOC Example
![image](https://github.com/gogo12235LYH/build-tfds/blob/master/images/voc.png)

### 如何讀取 ? tfds.load

注意 data_dir 位置選擇，若沒有設置會導向 C:\使用者\tensorflow_datasets\。

```python
import tensorflow_datasets as tfds

train, test = tfds.load(name="dpcb_db", split=["trainval", "test"], data_dir="D:\\tensorflow_datasets")
```

## 4. Tips

### tf.data: cache --> shuffle --> repeat --> batch --> prefetch

#### 1. Cache: 如果記憶體空間夠塞下整個訓練資料，可加入此方法
#### 2. Shuffle: 這裡會有一個 buffer size，也是取決記憶體大小設置(過小效果非常差，等同沒打亂)，會取每 shard 的前 buffer_size 個資料打亂
#### 2.1 Shuffle: 如果資料過大，建議包成 tfrecord 時，先將資料打亂；大型資料會有多個 shards，讀取時 shuffle_files = Ture，使每個 epoch 讀取的 shards 順序不同
#### 3. Repeat: 先重複再取 batch，確保每筆資料都能夠讀取到
#### 4. Batch: 如其名，但若是使用在多目標檢測上，要注意資料維度是否有補滿，使得 tensor 的 shape 都是固定的 !

```python
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
```