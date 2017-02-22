
## 參數的選擇

> <img src="http://www.nature.com/nbt/journal/v33/n8/images/nbt.3300-SF1.jpg" width="90%">
>... Shown is an example with __batch_size=5, motif_len=6, num_motifs=4, num_models=3__. Sequences are padded with ‘N’s so that the motif scan operation can find detections at both extremities. Yellow cells represent the reverse complement of the input located above; both strands are fed to the model, and the strand with the maximum score is used for the output prediction (the max strand stage). The output dimension of the pool stage, depicted as num_motifs (*), depends on whether “max” or “max and avg” pooling was used.

這篇論文其實測試了很多套不同的參數，所以只能試圖東拼西湊還原出最接近的模型。
論文中的motif detectors其實就是CNN的filter。 第十頁提到 motif_len 應選擇motif預期長度的1.5倍，但應為不知道應該預期 motif 長度是多少，所以最後選擇的長度是 slack 裡提示的 11 。寬度則是用第三頁範例裡的 m = 3。

batch_size則是參考上圖中提示的 batch_size=5。
// 更新：改batch_size為256後準確度大幅提升

depth根據第九頁的 num_motifs=16 選擇了16

關於NN的部分，文中提到最好只加一層32 units的hidden layer或根本不要加hidden layer的效果最好，在這邊選擇了前者。

故最後的Model選擇如下


## Model:
<pre>
           batch      height     width      channel
data:      5          121        4          1
conv:      5          121        4          16
relu:      5          121        4          16
pooling:   5          60         2          16
reshape:   5          1920
hidden:    5          32
output:    5          2
</pre>

## 參數：
<pre>
image_size   = [121,4]  ## 101bps, plus 10bps frenking on both end
num_labels   = 2        ## bind or not (1 or 0)
batch_size   = 256      ## TODO: try with double strand input!
filter_size  = [11,3]   ## Motif detector length = 11 (about 1.5 times of expected motif length)
depth        = 16       ## Number of motif detector (num_motif) = 16
num_hidden   = 32       ## 32 ReLU units of no hidden layer at all
num_steps    = 2000     ## 
</pre>

## test結果：72.9%
>
>	 Minibatch	 Minibatch	 Validation
Step	 Loss		 Accuracy	 Accuracy
0	 0.704212	 46.5%		 50.3%	
50	 0.711170	 50.0%		 51.7%	
100	 0.702906	 50.0%		 53.5%	
150	 0.681891	 53.9%		 55.6%	
200	 0.694169	 52.7%		 57.0%	
250	 0.661604	 60.5%		 52.5%	
300	 0.652424	 63.7%		 60.2%	
350	 0.648504	 60.9%		 58.5%	
400	 0.758930	 47.7%		 51.2%	
450	 0.606714	 69.9%		 61.2%	
500	 0.602666	 68.0%		 64.5%	
550	 0.666663	 55.1%		 63.6%	
600	 0.608256	 68.4%		 66.3%	
650	 0.617038	 64.8%		 67.7%	
700	 0.613045	 63.3%		 63.1%	
750	 0.659635	 58.6%		 59.8%	
800	 0.584620	 68.0%		 67.6%	
850	 0.565616	 70.7%		 67.7%	
900	 0.610997	 65.2%		 70.2%	
950	 0.587207	 72.3%		 70.8%	
1000	 0.595937	 65.6%		 68.7%	
1050	 0.650272	 62.9%		 63.7%	
1100	 0.549253	 73.0%		 71.5%	
1150	 0.647494	 62.5%		 65.4%	
1200	 0.531154	 73.0%		 71.1%	
1250	 0.545054	 73.0%		 70.8%	
1300	 0.604557	 70.7%		 67.9%	
1350	 0.572152	 69.1%		 70.9%	
1400	 0.535430	 70.7%		 68.5%	
1450	 0.517623	 73.8%		 71.8%	
1500	 0.623158	 64.8%		 70.5%	
1550	 0.529861	 73.0%		 73.4%	
1600	 0.554052	 71.1%		 72.7%	
1650	 0.602054	 64.8%		 66.2%	
1700	 0.559003	 70.3%		 72.6%	
1750	 0.520869	 76.2%		 72.5%	
1800	 0.537416	 74.2%		 72.5%	
1850	 0.587606	 64.8%		 68.4%	
1900	 0.552073	 69.5%		 72.7%	
1950	 0.624915	 61.7%		 65.2%	
2000	 0.531805	 71.1%		 71.7%	
*** TEST ACCURACY: 72.9% ***
>
