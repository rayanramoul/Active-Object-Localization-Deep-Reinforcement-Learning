# Active Object-Localization using Deep Reinforcement Learning
Considering object localization problem as a markov decision process and making an agent learn to maximize its reward on it.
Implementation based on paper : Caicedo, Juan  Lazebnik, Svetlana. (2015).”Active Object Localization with Deep Rein-forcement Learning”. 10.1109/ICCV.2015.286, <a href="https://arxiv.org/abs/1511.06015#:~:text=We%20present%20an%20active%20detection%20model%20for%20localizing%20objects%20in%20scenes.&text=This%20agent%20learns%20to%20deform,objects%20following%20top-down%20reasoning.">arxiv</a>

# Training results

<p float="left">
  <img src="https://github.com/raysr/Active-Object-Localization-Deep-Reinforcement-Learning/blob/master/media/ap_over_epochs.png" width="49%" />
  <img src="https://github.com/raysr/Active-Object-Localization-Deep-Reinforcement-Learning/blob/master/media/recall_over_epochs.png" width="49%" /> 
</p>



## Dataset and classes
Number of training elements per class in VOC2007 + VOC2012 <br>
cat : 648 elements.<br>
bird : 553 elements.       <br>
motorbike : 304 elements.  <br>
diningtable : 188 elements.<br>
train : 369 elements.      <br>
tvmonitor : 290 elements.  <br>
bus : 268 elements.        <br>
horse : 310 elements.      <br>
car : 659 elements.<br>
pottedplant : 202 elements.<br>
person : 1301 elements.<br>
chair : 379 elements.<br>
boat : 289 elements.<br>
bottle : 258 elements.<br>
bicycle : 303 elements.<br>
dog : 750 elements.<br>
aeroplane : 432 elements.<br>
cow : 210 elements.<br>
sheep : 208 elements.<br>
sofa : 297 elements.<br>



## Examples of Output
<p float="left">
  <img src="https://github.com/raysr/Active-Object-Localization-Deep-Reinforcement-Learning/blob/master/media/movie_0.gif" width="32%" />
  <img src="https://github.com/raysr/Active-Object-Localization-Deep-Reinforcement-Learning/blob/master/media/movie_1.gif" width="32%" /> 
  <img src="https://github.com/raysr/Active-Object-Localization-Deep-Reinforcement-Learning/blob/master/media/movie_2.gif" width="32%"/>
</p>


<p float="left">
<img src="https://github.com/raysr/Active-Object-Localization-Deep-Reinforcement-Learning/blob/master/media/movie_3.gif"  width="32%"/>
  <img src="https://github.com/raysr/Active-Object-Localization-Deep-Reinforcement-Learning/blob/master/media/movie_4.gif" width="32%" /> 
  <img src="https://github.com/raysr/Active-Object-Localization-Deep-Reinforcement-Learning/blob/master/media/movie_6.gif" width="32%" />
</p>


<p float="left">
<img src="https://github.com/raysr/Active-Object-Localization-Deep-Reinforcement-Learning/blob/master/media/movie_7.gif"  width="32%"/>
  <img src="https://github.com/raysr/Active-Object-Localization-Deep-Reinforcement-Learning/blob/master/media/movie_8.gif" width="32%" /> 
  <img src="https://github.com/raysr/Active-Object-Localization-Deep-Reinforcement-Learning/blob/master/media/movie_10.gif" width="32%" />
</p>


<p float="left">
<img src="https://github.com/raysr/Active-Object-Localization-Deep-Reinforcement-Learning/blob/master/media/movie_13.gif"  width="32%"/>
  <img src="https://github.com/raysr/Active-Object-Localization-Deep-Reinforcement-Learning/blob/master/media/movie_14.gif" width="32%" /> 
  <img src="https://github.com/raysr/Active-Object-Localization-Deep-Reinforcement-Learning/blob/master/media/movie_15.gif" width="32%" />
</p>


<p float="left">
<img src="https://github.com/raysr/Active-Object-Localization-Deep-Reinforcement-Learning/blob/master/media/movie_17.gif"  width="32%"/>
  <img src="https://github.com/raysr/Active-Object-Localization-Deep-Reinforcement-Learning/blob/master/media/movie_18.gif" width="32%" /> 
  <img src="https://github.com/raysr/Active-Object-Localization-Deep-Reinforcement-Learning/blob/master/media/movie_19.gif" width="32%" />
</p>



<p float="left">
<img src="https://github.com/raysr/Active-Object-Localization-Deep-Reinforcement-Learning/blob/master/media/movie_20.gif"  width="32%"/>
  <img src="https://github.com/raysr/Active-Object-Localization-Deep-Reinforcement-Learning/blob/master/media/movie_21.gif" width="32%" /> 
  <img src="https://github.com/raysr/Active-Object-Localization-Deep-Reinforcement-Learning/blob/master/media/movie_22.gif" width="32%" />
</p>


<p float="left">
<img src="https://github.com/raysr/Active-Object-Localization-Deep-Reinforcement-Learning/blob/master/media/movie_24.gif"  width="32%"/>
  <img src="https://github.com/raysr/Active-Object-Localization-Deep-Reinforcement-Learning/blob/master/media/movie_25.gif" width="32%" /> 
  <img src="https://github.com/raysr/Active-Object-Localization-Deep-Reinforcement-Learning/blob/master/media/movie_26.gif" width="32%" />
</p>

