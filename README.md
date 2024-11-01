# ToT-by-Scratch

## 介绍
这是一个即插即用的思维树（tree-of-thought）模块，参考了[官方实现](https://github.com/princeton-nlp/tree-of-thought-llm)，以及[第三方实现](https://github.com/kyegomez/tree-of-thoughts?tab=readme-ov-file)。

## 文件结构
- mothods 文件夹下包含具体的搜索算法，具备可拓展性，目前只实现了 BFS 算法。
- models 文件封装了调用大模型的相关接口以及思维树相关的 Prompt。

## 运行代码
```shell
cd methods.py
python bfs.py
```