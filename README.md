# sd-hypernetwork-merger
Hypernetwork を結合する。結合する Hypernetwork は以下の設定が一致している必要がある。  
This extension merges hypernetwork. The Hypernetwork to be combined must match the following settings.
1. Layer Structure
2. Layer Normalization
3. Dropout


以下の設定は一致していなくても結合できるが、動作は保証しない。Activation Function は Hypernetwork A のものが使われる。  
The following settings do not have to match to be combined, but operation is not guaranteed. The Activation Function is used for Hypernetwork A.  
1. Activation Function

## Activate Output
旧仕様では最終レイヤーに活性化関数が挿入されていた。これは意味がないので最新のバージョンでは使われていない。  
In the old specification, an activation function was inserted in the final layer. This makes no sense and is not used in the latest version.  

![](https://github.com/dskjal/sd-hypernetwork-merger/blob/main/misc/screenshot.png)
