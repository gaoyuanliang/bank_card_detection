# bank card tagging

Tagging images of road signs

<table>
  <thead>
    <tr>
      <th>Input</th>
      <th>Output</th>
    </tr>
  </thead>
  <tr>
    <td>
      <img src="https://www.mashreqbank.com/egypt/en/images/Platinum-Credit-Card_tcm74-220679.jpg" width="400">
    </td>
    <td>
      <pre>
{
  'tag': 'bank_card', 
  'score': 0.9966258
}
</pre>
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://ichef.bbci.co.uk/news/1024/media/images/67649000/jpg/_67649781_160757669.jpg" width="400">
    </td>
    <td>
      <pre>
{
  'tag': 'bank_card', 
  'score': 0.9973912
}
</pre>
    </td>
  </tr>
</table>

## instillation

to install the bank card detector, you need Python 3.7.7 

```bash
git clone https://github.com/gaoyuanliang/bank_card_detection.git

cd bank_card_detection

pip3 install -r requirements.txt

wget https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5
```

download the pretrain model of bank card detection from the following url

```bash
https://drive.google.com/file/d/10H70SUi1C5R79cu-27u-bg9ytnZ8JV9F/view?usp=sharing
```

to use the model to tag the image, we have four cases

## usage

**Test case #1**

download the test image by 

```bash
wget https://www.mashreqbank.com/egypt/en/images/Platinum-Credit-Card_tcm74-220679.jpg
```

the image looks like

<img src="https://www.mashreqbank.com/egypt/en/images/Platinum-Credit-Card_tcm74-220679.jpg" width="400">

run the test code

```python
from bank_card_detection import bank_card_detection

bank_card_detection('Platinum-Credit-Card_tcm74-220679.jpg')
```

you will see the following output since the image itself is a credit card

```python
{'tag': 'bank_card', 'score': 0.9966258}
```

**Test case #2**

download the test image by 

```bash
wget https://ichef.bbci.co.uk/news/1024/media/images/67649000/jpg/_67649781_160757669.jpg
```

the image looks like

<img src="https://ichef.bbci.co.uk/news/1024/media/images/67649000/jpg/_67649781_160757669.jpg" width="400">
 
run the test code

```python
bank_card_detection('_67649781_160757669.jpg')
```

you will see the following output since the image has a group of visa cards

```python
{'tag': 'bank_card', 'score': 0.9973912}
```

**Test case #3 and #4**

Then we will see two negative casese where the images have no bank card at all. 

Firstly download the images by 

```bash
wget https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Sample_cheque.jpeg/1200px-Sample_cheque.jpeg

wget https://www.uae-ix.net/Files/1e45b49e6debd4ed0a5e4dffbf17aa3fe5c04f89/Dubai_2_2340x800-1170x400.jpg
```

They look like 

 <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Sample_cheque.jpeg/1200px-Sample_cheque.jpeg" width="400">
 
<img src="https://www.uae-ix.net/Files/1e45b49e6debd4ed0a5e4dffbf17aa3fe5c04f89/Dubai_2_2340x800-1170x400.jpg" width="400">

Run the tagging codes

```python
bank_card_detection('1200px-Sample_cheque.jpeg')

bank_card_detection('Dubai_2_2340x800-1170x400.jpg')
```

You will see not outputs because there is no bank card at all

```python
{}

{}
```


### I am looking for job


feel free to contact me if you have any problem with this package or you are hiring data scientist/AI engineer. I am actively looking for data science/AI related jobs

My email: gaoyuanliang@outlook.com
