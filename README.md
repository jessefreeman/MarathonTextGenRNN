# MarathonTextGenRNN

This project is a proof of concept for using [textgenrnn](https://github.com/minimaxir/textgenrnn) to generate terminal story text for [Bungie's Marathon](https://en.wikipedia.org/wiki/Marathon_(video_game\)) game.

This is intended to not only show off the possibilities of training a model to create background stories but also show off some of the features [MissingLink.ai](https://missinglink.ai) provides when training models locally and in the cloud.

Here is an example of some text that was generated by a trained model:

```
* * * incoming message * * *

The Marathon is equipped with three artificial intelligence units. Tycho handles the science and engineering network, Durandal controls the marathon autonomous functions such as doors, air from kitchens, and I don't remember what to do.

The aliens seem to have been caught off guard by the strength of our counterattack. This is good news, but I have detected more ships landing on the marathon, and I fear that the aliens are reinforcing their efforts.

* * * end of message * * *
```

To run this project, use the [train.py](https://github.com/jessefreeman/MarathonTextGenRNN/blob/master/train.py) script to create a new model and the [generate.py](https://github.com/jessefreeman/MarathonTextGenRNN/blob/master/generate.py) script to output sample text. There is a dataset comprised of Marathon's terminal text scraped from the [official story site](http://marathon.bungie.org/story/contents.html).

You can use the [requirements.txt](https://github.com/jessefreeman/MarathonTextGenRNN/blob/master/requirements.txt) file to install the project's dependencies. There is also a [config.py](https://github.com/jessefreeman/MarathonTextGenRNN/blob/master/config.py) script you can use to modify settings during training and text generation.

## Credits

Max Woolf ([@minimaxir](http://minimaxir.com)) - [textgenrnn](https://github.com/minimaxir/textgenrnn)

*Max's open-source projects are supported by his [Patreon](https://www.patreon.com/minimaxir). If you found this project helpful, any monetary contributions to the Patreon are appreciated and will be put to good creative use.*

Andrej Karpathy for the original proposal of the char-rnn via the blog post [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).

[Daniel Grijalva](https://github.com/Juanets) for [contributing](https://github.com/minimaxir/textgenrnn/pull/52) an interactive mode.