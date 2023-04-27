---
title: DALLE-2
layout: post
---

<script
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
  type="text/javascript">
</script>


# DALLE-2 

This page will discuss the fundamentals and building blocks that are described in the DALLE-2 paper and model release. 

## Motivation and Background

We start by motivating the problem that DALLE-2 solves for: Image Generation. In A.I., we've seen various models that fall under the image generation objective:

- Generative Adversarial Networks (GANS)
- Autoencoders (AE) and Variational Autoencoders (VAEs)
- Denoising Diffusion Probabilistic Models (DDPMs)

But why would image generation be a useful task? Many might say that it is a well suited problem for ML models because the objective can be approached a well structured supervised learning experiments. In addition, this domain suits blending understandings of Natural Language and Computer Vision to understand how the two are understood by current ML models. From a commercial perspective, we see that there is a large demand for new content especially in the graphic design and digital art space and ML models make it seamless and extremely easy to quickly prototype or brainstorm designs for a variety of tasks whether it be logo generation, marketing content creation and the like. 

Now that we have some background of why this domain is of interest to ML researchers, we will provide some background on the methods that proved successful prior to the approach suggested for DALLE-2. 

### Generative Adversarial Networks

This section describes what is known as a Generative Adversarial Network or GAN. GANs are generally thought of as one of the first generative neural network models that were successfully applied to the image generation domain. The model was first introduced by Ian Goodfellow and his group in the paper [Generative Adversarial Networks [1]](https://arxiv.org/pdf/1406.2661.pdf). 

| ![Generative Adversarial Network Overview Diagram](assets/GAN.png) | 
|:--:| 
| *Generative Adversarial Network Diagram* taken from [[2]](https://arxiv.org/pdf/1710.07035.pdf) |

In this diagram, we see the general structure of the GAN based generative model. 

First we see the generator network `G` which is usually some sort of neural network structure that takes random noise `z` and generates an image `x'`. This is used as a training sample for the Discriminator network, also a similar neural network to the generator, but made with the objective of classifying any sample either from a real dataset `x` or from the generator `x'` to determine if the sample is fake or real. 

We can describe this structure a little bit more formally now:

- The Generator $$ G(z, \theta_g)$$ takes sample noise $$z$$ and uses the neural network (most likely Multilayer Perceptron (MLP)) parametrized by $$\theta_g$$ to generate the image sample $$x'$$

- The discriminator $$D(x, \theta_d)$$ takes any real sample image $$x$$ or generated sample image $$x'$$ and, classifies the image into 2 classes: real or fake. 

- The objective that the model is optimizing for can be described using the minimax objective:

$$ min_g max_d \: E_{x \sim p_{data}(x)}[log(D(x))] + E_{z \sim p_{G}(x)}[log(1-D(G(z)))]$$

We can see that this model is trying to maximize the loss of the Discriminator while the Generator tries to minimize the total loss. Lets take a look at the inital results that the paper presented for generative models. 

| ![GAN Results](assets/GAN_results.png) | 
|:--:| 
| *Generative Adversarial Network Outputs* taken from [[1]](https://arxiv.org/pdf/1406.2661.pdf) |

As we can see here, the first GANs were trained on widely known datasets and can be seen to produce fairly good output especially for simpler tasks like MNIST. 

We see a few limitations with GANs that did not allow them to become the most widely used generative model in the case of image generation when compared to other models in the domain:

1. Vanishing/Exploding gradient problem can cause the GAN network training to be very unstable. 

2. GANs require a very large dataset to model the data distribution. 

3. Because we are training 2 networks with different objectives, the training can also be very unstable for GANs. In this case, we have an algorithm provided by the original authors of the paper that uses SGD and partial training for updating the network. 

| ![GAN Training Algorithm](assets/GAN_algo.png) | 
|:--:| 
| *Generative Adversarial Network Training Algorithm* taken from [[1]](https://arxiv.org/pdf/1406.2661.pdf) |

By utilizing this algorithm, we can ensure much more stability in the training phase. 

With this model, we see the capabilities of ML models specifically deep neural network architectures to model image distributions and the high performance we get. Next we will talk about another useful class of models for image generations: Autoencoders (AE) with a special focus on the Variational Autoencoders. 

### Autoencoders and Variational Encoders

Autoencoders are another class of deep learning models that function very well at generative tasks. 

| ![AE Model Diagram](assets/Autoencoder_diag.png) | 
|:--:| 
| *Autoencoder Model Diagram* taken from [[3]](https://www.compthree.com/blog/autoencoder/) |

The general structure of an autoencoder model includes 2 neural network components: an Encoder and a Decoder. The Encoder architecture takes the original data and computes a lower dimensional representation of the data. The Decoder then takes this lower dimensional representation and computes an output for a variety of tasks, in this case the image that we would like to recreate as an input. 

These models are generally good at completing the generative task, but are much more suited for denoising approaches rather than generating new models as the distribution learned by the model for the latent vector is directly based on the training data. In this case, we will discuss a different architecture of Autoencoders that have been found to work really well for the generation task: Variational Autoencoders.

Variational Autoencoders (VAEs) introduced in the paper [Auto-Encoding Variational Bayes [6] ](https://arxiv.org/pdf/1312.6114.pdf#page=5)are essentially the same as autoencoders but they add another set of learnable parameters to the Encoder network that more directly try to represent the distribution of the latent space vectors that are being used to generate images in the decoder. 

| ![AE Model Diagram](assets/VAE_diag.png) | 
|:--:| 
| *Variational Autoencoder Model Diagram* taken from [[4]](https://medium.com/@judyyes10/generate-images-using-variational-autoencoder-vae-4d429d9bdb5) |

Here we see that the output of the Encoder network is passed through an additional 2 layers that represent the mean and variance. 

These additional layers represent the mean vector $$ \mu_z$$ and the standard deviation vector $$\sigma_z$$ of the latent space vector $$z$$. Together these parameters look to model a function $$q_{\phi}(z\mid x)$$ that is trying to fit the actual probability distribution $$p(z \mid x)$$ for getting a latent vector from any data in the distribution. Thus z would be sampled from $$ N(\mu, \sigma) $$ rather than a direct latent vector derived from the input sample.

One of the important tricks that was implemented in the VAE to be able to accurately train the network is the use of a "reparametrization trick" which allows the model to train the mean and standard deviation parameters. As we see with the diagram, $$z$$ is sampled from the Gaussian normal distribution $$N(\mu, \sigma)$$ but at the training phase, there is no way to propagate a gradient through a normal distribution sample as there is no gradient to pass back through these variables. Rather, we take sample a value $$\epsilon$$ from the isotropic gaussian $$N(0,I) $$ and use that random parameter to compute a latent vector $$z$$ that is still a part of the distribution but is computed using the equation we see in the diagram above: 

$$ z = \mu + \sigma \odot \epsilon $$

This way, we have set parameters that can be moved by the optimizer rather than just a sample that is derived from some normal distribution. This trick allows us to set the objective of the VAE to learn the mean and variance layers of the VAE. 

In the original paper, the loss function for the paper is described using the log likelihood function for the decoder and the KL loss for the encoder: 

$$ L(\theta, \phi, x) = E_{z \sim q_{\phi}(z \mid x)}[log \: p_{\theta}(x|z)] - KL(q_{\phi}(z \mid x) \| p_{\theta}(z)) $$

The first term is for the reconstruction loss of the decoder in recovering X from the latent sample, and the second term is to quantify how far the learned distribution $$q$$ is from the true distribution $$p_{\theta}(z)$$ of the latent vector space that is output by the encoder side of the VAE. In practicality, if the decoder loss $$p_{\theta}(x \mid z)$$ models a gaussian, this first term turns into an MSE loss. After some massaging of the equations, we can get a simplified loss function that looks like: 

$$ L(\theta, \phi, x) = - \frac{1}{2} \sum_{i} 1 + log \: \sigma_i - \mu^2_i - \sigma^2_i $$

Below we provide a look at some results of trainng the variational autoencoder model. As we can see, the VAE provides robust output for the objective of generating random human faces.


| ![AE Model Diagram](assets/VAE_faces.png) | 
|:--:| 
| *Variational Autoencoder Faces* taken from [[5]](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73) |

 When we look at the results, there are many examples of suitable or even photorealistic faces, but there are some that do not prove to be convincing outputs. In the case of VAEs, we see fairly good performance, but newer models like diffusion models have proven to give even better performance with similar training data requirements. 

### Denoising Diffusion Probabilistic Models

 Denoising Diffusion Probabilistric Models (DDPMs) are a class of deep learning models that are highly performant at the task of generating models. Similar to Autoencoders, they involve taking image and progressively adding gaussian noise to them until they reach a timestep which renders the image as complete gaussian noise. With these progressively noised samples, we train a model to try and predicts the previous timestep image with the noise removed. These two processes define a forward and backwards process for DDPMs. 

 | ![Diffusion Model Diagram](assets/Diffusion_diag.png) | 
|:--:| 
| *Diffusion Model Diagram* taken from [[6]](https://learnopencv.com/image-generation-using-diffusion-models/) |

Here we see the forwards process $$ q(x_t \mid x_{t-1})$$ which takes a pure image $$x_0$$ and produces the pure gaussian noise $$x_T$$ via the following function: 

$$ q(x_{1:T} \mid x_0) := \prod_{t=1}^T q(x_t \mid x_{t-1}) := \prod_{t=1}^T N(x_t, \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

In this case, Beta is a fixed variance schedule that provides us a pure Gaussian noised image at timestep $$T$$. 

The diffusion model is then actually learning the function $$p_{\theta}(x_{0:T})$$ in the backwards process which, as described before, learns the ability to go from $$x_t$$ back to $$x_0$$: 

$$ p_{\theta}(x_{0:T}) := \prod_{t=1}^T p(x_{t-1} \mid x_{t}) : \prod_{t=1}^T N(x_{t-1}, \mu_{\theta}( x_t,t), \sigma_{\theta}(x_t, t))$$

Similar to the VAE model, the diffusion model is learning parameters for the mean $$\mu_{\theta}( x_t,t)$$and variance $$ \sigma_{\theta}(x_t, t)$$ of the model when it is training to learn the backwards process. 

This class of models proves to be the best in photorealism as we see some examples below from Stable Diffusion, a widely known diffusion model that generates images from Natural Language prompts: 

| ![Diffusion Model Examples](assets/Diffusion_examples.png) | 
|:--:| 
| *Diffusion Model Examples* taken from [[7]](https://cdm.link/2022/08/stable-diffusion-the-slick-generative-ai-tool-just-launched-and-went-live-on-github/) |

The photorealism we see in these examples proves that diffusion models are a very strong class of models in generation of image samples. This model is a large basis of the generative part of DALLE-2 that gives the model its great photorealistic outputs. 


### Transformers

One model that should be given a small mention as it has become one of the most powerful and widely used ML machinery in the recent past is the Transformer. This model takes a sequence and computes a next most likely token for any sequence of inputs. Originally designed for the NLP task of sentence completion like the RNN and other NLP models before it, transformers proved to provide good output for a variety of other ML tasks like image classification and other generation tasks like image generation. 

The diagram below shows the structure of the Transformer model, here we see features of the Autoencoder structure with an encoder and a decoder structure within the transformer. 

| ![Transformer Model Structure](assets/Transformer_diag.png) | 
|:--:| 
| *Transformer Model Diagram* taken from [[8]](https://arxiv.org/pdf/1706.03762.pdf) |

This model utilizes a new module known as an Attention head, which computes the relevancy of one token in the data to any other token in the data. For more information on the calculation, and in the interest of saving time reading this article, please refer to Jay Alammar's [The Illustrated Transformer[9]](https://jalammar.github.io/illustrated-transformer/). 

This model is used largely in DALLE and DALLE-2 as a generative model, either for creating embeddings or directly for the generative task. 

### DALLE-1

An important part of image generation is the ability to condition the output of the model on natural language captions that describe what is desired of the output of the model. The first DALLE model was created by OpenAI to approach this task. Described in the paper [Zero-Shot Text-to-Image Generation[10]](https://arxiv.org/pdf/2102.12092.pdf). This model utilizes a few components for image generation that can be captioned on the :

1. The 256x256 image tokens that are used as training data for the generative model are encoded into 32x32 grid of tokens which can each assume 8192 values that are encoded by an discrete Variational Autoencoder. 

2. The image captions are encoded using Byte-Pair Encoding (BPE) which takes the most common bytes in a sequence and replaces them with a byte that is very uncommon which compresses the data significantly. This allows the most common phrases are represented as a single token as opposed to multiple tokens. These tokens are concatenated with the image tokens and passed to a transformer to model the joint distribution of the text and image tokens.

In this case, GPT-3 is used as the transformer that autoregressively models the text and image captions that can be used by the dVAE decoder to generate real images. 

| ![DALLE Model Results](assets/DALLE_results.png) | 
|:--:| 
| *DALLE-1 Model Results* taken from [[11]](https://arxiv.org/pdf/2112.10741v1.pdf) |

When we take a look at the results of DALLE, the model performs fairly well at generating images that match the text, but the image quality and photorealism that is seen by other models like the DDPMs (in this case OpenAI's GLIDE diffusion model) or other GANs prove to be less than overwhelming. As a result, we need something that can handle the text-conditional image generation task with photorealistic output. In comes DALLE-2...

## DALLE-2

The focus of this article is the DALLE-2 model. We have spent a large part of the article describing various other models that all build to this model. The model is described in the paper [Hierarchical Text-Conditional Image Generation with CLIP Latents[12]](https://cdn.openai.com/papers/dall-e-2.pdf). 

There are 2 major parts of the model that are used in creating the images based on the desired caption for the image: CLIP and unClip, which is made up of a prior model and the decoder. CLIP is a separate model that takes images and text and encodes both the image and text such that images with features similar to the relevant caption are in the same space. The prior model takes a caption and samples a CLIP embedding similar to the caption such that the features from the caption would be represented in the image. Finally the 

| ![DALLE-2 Model Diagram](assets/DALLE_2_diag.png) | 
|:--:| 
| *DALLE-2 Model Diagram* taken from [[13]](https://cdn.openai.com/papers/dall-e-2.pdf) |

First we will discuss CLIP as it is a powerful model for creating joint embeddings between images and text in the same vector space. 

### CLIP

Contrastive Learning Image Pretraining or CLIP is an OpenAI model introduced in [Learning Transferable Visual Models From Natural Language Supervision[14]](https://arxiv.org/pdf/2103.00020.pdf). As described before, the objective of the CLIP model is to learn a joint embedding space that places images and text together. The model is pretrained on the task of learning which caption works best with any image which proves to be very efficient in then allowing the model to then model downstream tasks like image classification in a zero shot nature because of the visual features learned by training the model on captions for images. 

As we can see in the pretraining phase, we have 2 encoders which encode the images and text into embeddings which are then multiplied together to get a score. The model optimizes the embedding such that the text and image embeddings are maximized when the caption fits the image. This allows the model to learn what words actually look like in images and vice-versa. 

| ![CLIP Model Diagram](assets/CLIP_diag.png) | 
|:--:| 
| *CLIP Model Diagram* taken from [[14]](https://arxiv.org/pdf/2103.00020.pdf) |

In our case for DALLE-2 we are mainly interested in the embeddings generated by the trained model which represent the similarity between images and captions and represent visual features that are defined by natural language captions. 

In the paper there is very simple example of the code that would be required for CLIP that shows how simple yet powerful a model can be at scale: 

| ![CLIP Code ](assets/CLIP_code.png) | 
|:--:| 
| *CLIP Pseudocode Description* taken from [[14]](https://arxiv.org/pdf/2103.00020.pdf) |

With CLIP, we will be able to generate embeddings for both text that represent images we would like or take image embeddings for the captions we would like. 

### unCLIP

In the second part of the model, described as unCLIP by the OpenAI paper, we see that this model utilizes two models for the purpose of taking the actual text caption that is being requested and generating the image that is requested. The model assumes that we are given pairs of $$(x,y)$$ with images $$x$$ and captions $$y$$. Next we have the image embeddings $$z_i$$ and $$z_t$$ that are generated from CLIP. As mentioned before, there are 2 parts of the model that we will discuss formally.

- A prior $$P(z_i \mid y)$$ that produces CLIP image embeddings based on the caption. 
- A decoder $$P(x \mid z_i, y)$$ that generates an image given a CLIP embedding which can also be conditioned on the text caption given to unCLIP. 

These models come together to form a generative model:

$$P(x \mid y) = P(x, z_i \mid y) = P(x \mid z_i, y) P(z_i \mid y) $$

As we can see, the generative model that gives an image $$x$$ based on the caption $$y$$ decomposes into our 2 components which allows us to have high performance and diversity when we model these probabilities well. 

### Prior 

In the paper there are 2 prior models tested: an autoregressive model and a diffusion model.

In the case of the Autoregressive prior, we utilize a transformer based model to predict the CLIP image embedding $$z_i$$ based on the caption $$y$$. The diffusion model prior takes $$z_i$$ and models it using the diffusion forward and backwards process while conditioned on caption $$y$$. In the interest of brevity, the specifics of the methods used for each method can be seen in the paper, but it is worth mentioning that the group decided that the diffusion model would be the more suitable solution given that it provided similar performance with much more computation.  

### Decoder

The decoder, which generates the image given the CLIP embedding $$z_i$$ which is designed by using another OpenAI model called GLIDE. This model is based on diffusion models, but conditions the foward process of the model on the output. 

| ![GLIDE Diagram](assets/CLIP_code.png) | 
|:--:| 
| *GLIDE Model Diagram* taken from [[15]](https://www.assemblyai.com/blog/how-dall-e-2-actually-works/) |

As we can see, the caption is passed to a transformer and the final token is used as a conditional token for the diffusion model to use in the noising process. This bakes in the caption to the training data that the backwards process model learns on. This provides high performance as the decoder becomes conditionally aware of the tokens that were used to describe the noisy image. 

Another aspect of bringing the photorealistic part of the image generation task is to upsample the images using diffusion models as well. The output of the decoder is in 64x64 which is then upsampled to 256x256 and then 1024x1024. 

Now when we put all these together, we are able to prompt DALLE-2 for any caption and get very photorealistic results from them as seen below with an example directly from the paper. 

| ![DALLE-2 Results](assets/DALLE_2_sample.png) | 
|:--:| 
| *DALLE-2 Samples * taken from [[12]](https://cdn.openai.com/papers/dall-e-2.pdf) |

## Remarks

As we have seen, there are many developments in Generative AI and Image Generation that were required in bringing DALLE-2 to fruition. Without these developments, it would not have been possible to model the probabilities that are used by unCLIP to generate images from captions. When we look at the discussions of DALLE-2, there are a few limitations like its ability to understand semantics within captions as seen in the image below. 


| ![DALLE-2 Errors](assets/DALLE_2_errors.png) | 
|:--:| 
| *DALLE-2 Problems with Semantic Captions * taken from [[12]](https://cdn.openai.com/papers/dall-e-2.pdf) |

However these limitations could prove to be removed with stronger representations within the CLIP embedding spaces. As we allow the representational space that DALLE-2 uses, in this case CLIP, to become robust to things like orientation and specific features, we will find that the model will be good at even producing images for very complex captions, which would prove that we can model text and images in a joint space at a very high level. 

One thing that I would suggest more than anything after reading this article is to try the model out yourself! Feel free to check it out at [OpenAI](https://labs.openai.com/)