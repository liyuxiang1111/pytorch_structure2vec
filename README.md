# pytorch_structure2vec
使用pytorch实现的structure2vec
pytorch implementation of structure2vec

## Requirements
以下版本已通过测试，但更新的版本应该也可以使用：

- rdkit: [2017年第三季度发布](https://github.com/rdkit/rdkit/releases/tag/Release_2017_09_1, Release_2017_09_2)
- boost: Boost 1.61.0, 1.65.1
The following versions have been tested. But newer versions should also be fine. 

- rdkit : [Q3 2017 Release](https://github.com/rdkit/rdkit/releases/tag/Release_2017_09_1, Release_2017_09_2)
- boost : Boost 1.61.0, 1.65.1

## Setup
构建c++后端，然后一切就绪
Build the c++ backend of s2v_lib and you are all set.

```
cd s2v_lib
make -j4  
```

## Reproduce Experiments on Harvard Clean Energy Project
首先你需要从源码安装 rdkit (https://github.com/rdkit/rdkit)。然后，将环境变量 RDBASE 设置为你构建的 rdkit 的路径。
First, you need to install rdkit (https://github.com/rdkit/rdkit) from source. Then set RDBASE to your built rdkit.
```
export RDBASE=/path/to/your/rdkit
```
构建 harvard_cep 的 C++ 后端。
Build the c++ backend of harvard_cep. 

```
cd harvard_cep
make -j4
```

### Prepare data
意思是原始数据和处理后的数据可以通过以下链接获取：
https://www.dropbox.com/sh/eylta6a24fc9xo4/AAANyIgKnq49HB0Ud989JGEZa?dl=0

下载文件后，将它们放到 data 文件夹中。

我们在论文中使用了相同的数据集 (Dai 等人, ICML 2016)。这里的数据划分是 Wengong Jin 提供的，链接在 Google Drive。因此，可能会观察到一些轻微的性能提升。

The raw data and cooked data are available at the following link:
https://www.dropbox.com/sh/eylta6a24fc9xo4/AAANyIgKnq49HB0Ud989JGEZa?dl=0

After you download the files, put them under the data folder. 

We used the same dataset in our paper (Dai. et.al, ICML 2016). Here the data split as is provided by [Wengong Jin](http://people.csail.mit.edu/wengong/) in [google drive](https://drive.google.com/drive/folders/0B0GLTTNiVPEkdmlac2tDSzBFVzg). So minor performance improvement is observed. 

##### cook data
上面的 Dropbox 文件夹已经包含了处理好的数据。但是，如果你想自己处理数据，只需要将原始的 txt 数据下载到 data 文件夹中，然后执行以下操作即可。
The above dropbox folder already contains the cooked data. But if you want to cook it on your own, then you just need to download the raw txt data into the data folder, and do the following:

```
cd harvard_cep
python mol_lib.py
```

### Model dump
预训练模型保存在saved/文件夹下
The pretrained model is under ```saved/``` folder. 

##### for mean_field: 
```
$ python main.py -gm mean_field -saved_model saved/mean_field.model -phase test
====== begin of s2v configuration ======
| msg_average = 0
======   end of s2v configuration ======
loading data
train: 1900000
valid: 82601
test: 220289
loading model from saved/epoch-best.model
loading graph from data/test.txt.bin
num_nodes: 6094162	num_edges: 7357400
100%|███████████████████████████████████████████████████████████████████████████████████| 220289/220289 [00:01<00:00, 130103.34it/s]
mae: 0.08846 rmse: 0.11290: 100%|███████████████████████████████████████████████████████████| 4406/4406 [00:15<00:00, 279.01batch/s]
average test loss: mae 0.07017 rmse 0.09724
```
##### for loopy_bp:
```
$ python main.py -gm loopy_bp -saved_model saved/loopy_bp.model -phase test
====== begin of s2v configuration ======
| msg_average = 0
======   end of s2v configuration ======
loading data
train: 1900000
valid: 82601
test: 220289
loading model from saved/loopy_bp.model
loading graph from data/test.txt.bin
num_nodes: 6094162	num_edges: 7357400
100%|███████████████████████████████████████████████████████████████████████████████████| 220289/220289 [00:01<00:00, 131913.93it/s]
mae: 0.06883 rmse: 0.08762: 100%|███████████████████████████████████████████████████████████| 4406/4406 [00:17<00:00, 246.84batch/s]
average test loss: mae 0.06212 rmse 0.08747

```

#### Reference

```bibtex
@article{dai2016discriminative,
  title={Discriminative Embeddings of Latent Variable Models for Structured Data},
  author={Dai, Hanjun and Dai, Bo and Song, Le},
  journal={arXiv preprint arXiv:1603.05629},
  year={2016}
}
```
