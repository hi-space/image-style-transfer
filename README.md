# Style Transfer AI

## Requirement

### Install

```sh
torch==1.4.0
torchvision==0.5.0
torchsummary
Pillow
```

### Dataset

```sh
wget http://images.cocodataset.org/zips/val2017.zip
```

- Download the COCO dataset
- Make `coco` directory and put it in

## Train

```py
python train.py
```

Check the setting in the `train_config.py` file before training.

- `style_image_location` : Set the path to the image of the style you want
- `checkpoint_dir` : Directory where checkpoints are saved
- `transfer_learning`, `ckpt_model_path` : If you have any checkpoint, you can use the transfer learning.

## Test

Check the setting in the `test_config.py` file before testing.

- `checkpoint_dir`, `checckpoint_file` : Set a trained style checkpoint

### Test Image

```py
python test_image.py
```

- `test_image` : Image path to inference
- `output_image` : Inferenced image path

### Test Video

```py
python test_video.py
```

- `source_file` : Video path to inference
- `output_file` : Inferenced video path
- `debug_dir` : Path to save the inferenced result for each frame

---

## Result Video

[![Image Style Transfer using CNN](http://img.youtube.com/vi/KvjJ5BLa058/0.jpg)](https://youtu.be/8BvFx-VpFQU)

---

## Image Style

| Artist | Title | Style | Result |
|--------|-------|-------|--------|
|   Claude Monet     |  Charing Cross Bridge     |  ![monet-charing_cross_bridge](./image/monet-charing_cross_bridge.jpg)     |   ![monet-charing_cross_bridge](./image/monet-charing_cross_bridge-example.png)     |
|   Claude Monet     |   San Giorgio Maggiore at Dusk    |   ![monet-san_giorgio_maggiore_at_dusk](./image/monet-san_giorgio_maggiore_at_dusk.jpg)    |    ![monet-san_giorgio_maggiore_at_dusk](./image/monet-san_giorgio_maggiore_at_dusk-example.png)    |
| Vincent van Gogh       |  Bedroom in Arles      |  ![vangogh-bedroom_in_arles](./image/vangogh-bedroom_in_arles.jpg)     | ![vangogh-bedroom_in_arles](./image/vangogh-bedroom_in_arles-example.png)       |
| Vincent van Gogh       | Starry Night      |  ![vangogh-starry_night](./image/vangogh-starry_night.png)      |  ![vangogh-starry_night](./image/vangogh-starry_night-example.png)       |
| Vincent van Gogh       | Starry Night Over the Rhone      |  ![vangogh-starry_night_over_the_rhone](./image/vangogh-starry_night_over_the_rhone.png)     |  ![vangogh-starry_night_over_the_rhone](./image/vangogh-starry_night_over_the_rhone-example.png)      |
|   Pablo Picasso     |  Studio with Plaster Head     |    ![picasso-studio_with_plaster_head](./image/picasso-studio_with_plaster_head.jpg)    |   ![picasso-studio_with_plaster_head](./image/picasso-studio_with_plaster_head-example.png)      |
| Edgar Degas       | The Dance Foyer at the Opera on the rue Le Peletier      |   ![degas-the_dance_foyer_at_the_opera](./image/degas-the_dance_foyer_at_the_opera.jpg)    |  ![degas-the_dance_foyer_at_the_opera](./image/degas-the_dance_foyer_at_the_opera-example.png)         |
|  Henri de Toulouse      |   In bed    |  ![toulouse-in_bed](./image/toulouse-in_bed.png)      |  ![toulouse-in_bed](./image/toulouse-in_bed-example.png)       |
|  Paul Gauguin      |  Te Fare (La maison)     | ![gauguin-te_fare](./image/gauguin-te_fare.jpg)       |  ![gauguin-te_fare](./image/gauguin-te_fare-example.png)      |

## Examples

### Claude Monet

#### Charing Cross Bridge

| Style  | Result |
|--------|--------|
| ![monet-charing_cross_bridge](./image/monet-charing_cross_bridge.jpg)       |   ![monet-charing_cross_bridge](./image/monet-charing_cross_bridge-examples.png)     |

#### San Giorgio Maggiore at Dusk

| Style | Result |
|-------|--------|
| ![monet-san_giorgio_maggiore_at_dusk](./image/monet-san_giorgio_maggiore_at_dusk.jpg)      |    ![monet-san_giorgio_maggiore_at_dusk](./image/monet-san_giorgio_maggiore_at_dusk-examples.png)    |

### Vincent van Gogh

#### Bedroom in Arles

| Style | Result |
|-------|--------|
| ![vangogh-bedroom_in_arles](./image/vangogh-bedroom_in_arles.jpg)      |   ![vangogh-bedroom_in_arles](./image/vangogh-bedroom_in_arles-examples.png)     |

#### Starry Night

| Style | Result |
|-------|--------|
|  ![vangogh-starry_night](./image/vangogh-starry_night.png)     |  ![vangogh-starry_night](./image/vangogh-starry_night-examples.png)      |

#### Starry Night Over the Rhone

| Style | Result |
|-------|--------|
|  ![vangogh-starry_night_over_the_rhone](./image/vangogh-starry_night_over_the_rhone.png)     |   ![vangogh-starry_night_over_the_rhone](./image/vangogh-starry_night_over_the_rhone-examples.png)     |

### Pablo Picasso

#### Studio with Plaster Head

| Style | Result |
|-------|--------|
|  ![picasso-studio_with_plaster_head](./image/picasso-studio_with_plaster_head.jpg)     |  ![picasso-studio_with_plaster_head](./image/picasso-studio_with_plaster_head-examples.png)      |

### Edgar Degas

#### The Dance Foyer at the Opera on the rue Le Peletier

| Style | Result |
|-------|--------|
|   ![degas-the_dance_foyer_at_the_opera](./image/degas-the_dance_foyer_at_the_opera.jpg)    |  ![degas-the_dance_foyer_at_the_opera](./image/degas-the_dance_foyer_at_the_opera-examples.png)      |

### Henri de Toulouse

#### In bed

| Style | Result |
|-------|--------|
| ![toulouse-in_bed](./image/toulouse-in_bed.png)      |   ![toulouse-in_bed](./image/toulouse-in_bed-examples.png)     |

### Paul Gauguin

#### Te Fare (La maison)

| Style | Result |
|-------|--------|
|  ![gauguin-te_fare](./image/gauguin-te_fare.jpg)     |  ![gauguin-te_fare](./image/gauguin-te_fare-examples.png)      |

## Reference

- [Image Style Transfer Using Convolutional Neural Networks (2016, ECCV)](https://arxiv.org/pdf/1603.08155.pdf)
- [fast_neural_style](https://github.com/pytorch/examples/tree/master/fast_neural_style)
