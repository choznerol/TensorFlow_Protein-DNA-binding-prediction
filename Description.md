
## 參數的選擇

> <img src="http://www.nature.com/nbt/journal/v33/n8/images/nbt.3300-SF1.jpg" width="90%">
>... Shown is an example with __batch_size=5, motif_len=6, num_motifs=4, num_models=3__. Sequences are padded with ‘N’s so that the motif scan operation can find detections at both extremities. Yellow cells represent the reverse complement of the input located above; both strands are fed to the model, and the strand with the maximum score is used for the output prediction (the max strand stage). The output dimension of the pool stage, depicted as num_motifs (*), depends on whether “max” or “max and avg” pooling was used.

這篇論文其實測試了很多套不同的參數，所以只能試圖東拼西湊還原出最接近的模型。
論文中的motif detectors其實就是CNN的filter。 第十頁提到 motif_len 應選擇motif預期長度的1.5倍，但應為不知道應該預期 motif 長度是多少，所以最後選擇的長度是 slack 裡提示的 11 。寬度則是用第三頁範例裡的 m = 3。

batch_size則是參考上圖中提示的 batch_size=5。

depth根據第九頁的 num_motifs=16 選擇了16

關於NN的部分，文中提到最好只加一層32 units的hidden layer或根本不要加hidden layer的效果最好，在這邊選擇了前者。

故最後的Model選擇如下


<code>
## Model:
           [batch, height, width, channel]
data:      [5, 121, 4, 1]
conv:      [5, 121, 4, 16]
relu:      [5, 121, 4, 16]
pooling:   [5, 60, 2, 16]
reshape:   [5, 1920]
hidden:    [5, 32]
output:    [5, 2]


## 參數：

image_size   = [121,4]  ## 101bps, plus 10bps frenking on both end
num_labels   = 2        ## bind or not (1 or 0)
batch_size   = 5        ## TODO: try with double strand input!
filter_size  = [11,3]   ## Motif detector length = 11 (about 1.5 times of expected motif length)
depth        = 16       ## Number of motif detector (num_motif) = 16
num_hidden   = 32       ## 32 ReLU units of no hidden layer at all

</code>


> There are four computational stages, in order: convolution, rectification, pooling, and neural network.

但說了這麼多，目前test結果還是淒慘的50% QQ，原以為做到論文中提到的四個步驟就可以，結果留給turing的時間太少了也還沒跟其他人討論過，所以接下來還要繼續tune才行，首先應該會先嘗試加入early stopping


