# Bank card image tagging by deep learning

Tagging images of bank cards, such as credit card, debit card, etc, based on Xception pretrained deep feature extraction and my own trained classification layers

<img src="https://github.com/gaoyuanliang/bank_card_detection/raw/master/Webp.net-gifmaker.gif" width="600">

Same image tagging examples: 

<table>
  <thead>
    <tr>
      <th>Input</th>
      <th>Output</th>
    </tr>
  </thead>
  <tr>
    <td>
      <img src="https://github.com/gaoyuanliang/bank_card_detection/blob/master/bank_card.jpg" width="400">
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
      <img src="https://github.com/gaoyuanliang/bank_card_detection/blob/master/non_bank_card.jpeg" width="400">
    </td>
    <td>
      <pre>
{
  'tag': 'non_bank_card', 
  'score': 0.99998796
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

to use the model to tag the image, we have three cases

## usage

**Test case #1**

the first image looks like

<img src="https://github.com/gaoyuanliang/bank_card_detection/blob/master/bank_card.jpg" width="400">

run the test code

```python
from bank_card_detection import bank_card_detection

bank_card_detection('bank_card.jpg')
```

you will see the following output since the image itself is a credit card

```python
{'tag': 'bank_card', 'score': 0.9966258}
```

**Test case #2**

the seconde image looks like

<img src="https://github.com/gaoyuanliang/bank_card_detection/blob/master/bank_card1.jpg" width="400">
 
run the test code

```python
bank_card_detection('bank_card1.jpg')
```

you will see the following output since the image has a group of visa cards

```python
{'tag': 'bank_card', 'score': 0.9973912}
```

**Test case #3**

Then we will see one negative casese where the image have no bank card at all. The image looks like 

<img src="https://github.com/gaoyuanliang/bank_card_detection/blob/master/non_bank_card.jpeg" width="400">
 
Run the tagging codes

```python
bank_card_detection('non_bank_card.jpeg')
```

You will see the output of non bank card because there is no bank card at all

```python
{'tag': 'non_bank_card', 'score': 0.99998796}
```

### I am looking for job


feel free to contact me if you have any problem with this package or you are hiring data scientist/AI engineer. I am actively looking for data science/AI related jobs

My email: yanliang2345@outlook.com
