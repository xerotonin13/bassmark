# :loud_sound: bassmark: Proactive Localized Watermarking

Inference code for AudioSeal, a method for speech localized watermarking, with state-of-the-art robustness and detector speed (training code coming soon).
More details can be found in the [paper](https://arxiv.org/abs/2401.17264).

# Abtract

# :mate: Installation

```
pip install audioseal
```

To install from source: Clone this repo and install in editable mode:

```
git clone https://github.com/facebookresearch/audioseal
cd audioseal
pip install -e .
```

# :gear: Models

You can find all the model checkpoints on the [Hugging Face Hub](https://huggingface.co/facebook/audioseal). We provide the checkpoints for the following models:

- [AudioSeal Generator](src/audioseal/cards/audioseal_wm_16bits.yaml).
  It takes as input an audio signal (as a waveform), and outputs a watermark of the same size as the input, that can be added to the input to watermark it.
  Optionally, it can also take as input a secret message of 16-bits that will be encoded in the watermark.
- [AudioSeal Detector](src/audioseal/cards/audioseal_detector_16bits.yaml).
  It takes as input an audio signal (as a waveform), and outputs a probability that the input contains a watermark at each sample of the audio (every 1/16k s).
  Optionally, it may also output the secret message encoded in the watermark.

Note that the message is optional and has no influence on the detection output. It may be used to identify a model version for instance (up to $2**16=65536$ possible choices).

**Note**: We are working to release the training code for anyone wants to build their own watermarker. Stay tuned !

# :abacus: Usage

Audioseal provides a simple API to watermark and detect the watermarks from an audio sample. Example usage:

```python

from audioseal import AudioSeal

# model name corresponds to the YAML card file name found in audioseal/cards
model = AudioSeal.load_generator("audioseal_wm_16bits")

# Other way is to load directly from the checkpoint
# model =  Watermarker.from_pretrained(checkpoint_path, device = wav.device)

# a torch tensor of shape (batch, channels, samples) and a sample rate
# It is important to process the audio to the same sample rate as the model
# expectes. In our case, we support 16khz audio 
wav, sr = ..., 16000

watermark = model.get_watermark(wav, sr)

# Optional: you can add a 16-bit message to embed in the watermark
# msg = torch.randint(0, 2, (wav.shape(0), model.msg_processor.nbits), device=wav.device)
# watermark = model.get_watermark(wav, message = msg)

watermarked_audio = wav + watermark

detector = AudioSeal.load_detector("audioseal_detector_16bits")

# To detect the messages in the high-level.
result, message = detector.detect_watermark(watermarked_audio, sr)

print(result) # result is a float number indicating the probability of the audio being watermarked,
print(message)  # message is a binary vector of 16 bits


# To detect the messages in the low-level.
result, message = detector(watermarked_audio, sr)

# result is a tensor of size batch x 2 x frames, indicating the probability (positive and negative) of watermarking for each frame
# A watermarked audio should have result[:, 1, :] > 0.5
print(result[:, 1 , :])  

# Message is a tensor of size batch x 16, indicating of the probability of each bit to be 1.
# message will be a random tensor if the detector detects no watermarking from the audio
print(message)  
```

# Train your own watermarking model

See [here](./docs/TRAINING.md) for details on how to train your own Watermarking model.

# Want to contribute?

 We welcome Pull Requests with improvements or suggestions.
 If you want to flag an issue or propose an improvement, but dont' know how to realize it, create a GitHub Issue.

# Troubleshooting

- If you encounter the error `ValueError: not enough values to unpack (expected 3, got 2)`, this is because we expect a batch of audio  tensors as inputs. Add one
dummy batch dimension to your input (e.g. `wav.unsqueeze(0)`, see [example notebook for getting started](examples/Getting_started.ipynb)).

- In Windows machines, if you encounter the error `KeyError raised while resolving interpolation: "Environmen variable 'USER' not found"`: This is due to an old checkpoint
uploaded to the model hub, which is not compatible in Windows. Try to invalidate the cache by removing the files in `C:\Users\<USER>\.cache\audioseal`
and re-run again.

- If you use torchaudio to handle your audios and encounter the error `Couldn't find appropriate backend to handle uri ...`, this is due to newer version of
torchaudio does not handle the default backend well. Either downgrade your torchaudio to `2.1.0` or earlier, or install `soundfile` as your audio backend.

# License

- The code in this repository is released under the MIT license as found in the [LICENSE file](LICENSE).

# Maintainers:
- [Tuan Tran](https://github.com/antoine-tran)
- [Hady Elsahar](https://github.com/hadyelsahar)
- [Pierre Fernandez](https://github.com/pierrefdz)
- [Robin San Roman](https://github.com/robinsrm)

