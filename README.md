# sd-hypernetwork-merger
Hypernetwork を結合する。結合する Hypernetwork は以下の設定が一致している必要がある。  
This extension merges hypernetwork. The Hypernetwork to be combined must match the following settings.
1. Layer Structure
2. Layer Normalization

　  
以下の設定は一致していなくても結合できるが、動作は保証しない。Activation Function と Dropout の設定は Hypernetwork A のものが使われる。  
The following settings do not have to match to be combined, but operation is not guaranteed. The Activation Function and Dropout settings are used for Hypernetwork A.  
1. Activation Function
2. Dropout

![](https://github.com/dskjal/sd-hypernetwork-merger/blob/main/misc/screenshot.png)
