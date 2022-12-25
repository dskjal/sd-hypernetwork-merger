# sd-hypernetwork-merger
Hypernetwork を結合する。結合する Hypernetwork は以下の設定が一致している必要がある。  
This extension merges hypernetwork. The Hypernetwork to be combined must match the following settings.
- Layer Structure
- Layer Normalization
- Dropout


以下の設定は一致していなくても結合できるが、動作は保証しない。Activation Function は Hypernetwork A のものが使われる。  
The following settings do not have to match to be combined, but operation is not guaranteed. The Activation Function is used for Hypernetwork A.  
- Activation Function

## Last Layer Dropout
旧仕様では最後の全結合層の後にも dropout が挿入されていた。最新バージョンでは最後の全結合層の後には dropout は挿入しない。  
In the old specification, a dropout was also inserted after the last linear layer. In the latest version, the dropout is not inserted after the last linear layer.

## Activate Output
旧仕様では最終レイヤーに活性化関数が挿入されていた。これは意味がないので最新のバージョンでは使われていない。  
In the old specification, an activation function was inserted in the final layer. This makes no sense and is not used in the latest version.  

## [Module 1024](https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/1123f52cadf8d86c006177791b3191e5b8388b5a)
新しい OpenCLIP を使う場合に有効にする。  
Enable when using the new OpenCLIP.

![](https://github.com/dskjal/sd-hypernetwork-merger/blob/main/misc/screenshot.png)
