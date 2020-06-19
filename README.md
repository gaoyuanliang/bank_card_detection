![alt text](https://www.mashreqbank.com/egypt/en/images/Platinum-Credit-Card_tcm74-220679.jpg)



to install the cheque detector model, you need Python 3.7.7 

> git clone https://github.com/gaoyuanliang/bank_card_detection.git
>
> cd bank_card_detection
>
> pip3 install -r requirements.txt
>
> wget https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5

download the pretrain model of bank card detection from the following url

> https://drive.google.com/file/d/10H70SUi1C5R79cu-27u-bg9ytnZ8JV9F/view?usp=sharing

to use the model to tag the image



run the test code

> from cheque_detection import cheque_detection
>
> print(cheque_detection("test_sample1.jpg"))
> 
> print(cheque_detection("test_sample2.jpg"))
>
> print(cheque_detection("test_sample3.jpg"))



