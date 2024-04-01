import numpy as np
import tensorflow as tf
import pandas as pd
from tpae import vis

def prepare(patches, patch_meta, sampleid_name='id', frac_train=0.8, n_examples=16):
    sample_onehot = pd.get_dummies(patch_meta[sampleid_name]).astype('float32')
    sampleids = sample_onehot.columns
    patch_meta = pd.concat([patch_meta, sample_onehot], axis=1)

    ind = np.argsort(np.random.randn(patches.shape[0]))
    train = ind[:int(frac_train*len(ind))]
    test = ind[len(train):]

    x_train = np.ascontiguousarray(patches[train])
    sids_train = np.ascontiguousarray(patch_meta.loc[train, sampleids].values)
    x_test = np.ascontiguousarray(patches[test])
    sids_test = np.ascontiguousarray(patch_meta.loc[test, sampleids].values)

    x_test_examples = x_test[:n_examples]
    sids_test_examples = sids_test[:n_examples]

    return patch_meta, sampleids, x_train, sids_train, x_test, sids_test, x_test_examples, sids_test_examples

@tf.keras.saving.register_keras_serializable()
class AE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim, ncolors, nsamples, patchsize):
        super(AE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
            tf.keras.layers.InputLayer(input_shape=(patchsize, patchsize, ncolors)),
            tf.keras.layers.Conv2D(
                filters=256, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=512, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
            )
        
        self.decoder = tf.keras.Sequential(
            [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=(patchsize//4)*(patchsize//4)*512, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(patchsize//4, patchsize//4, 512)),
            tf.keras.layers.Conv2DTranspose(
                filters=256, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=ncolors, kernel_size=3, strides=2, padding='same'),
            ]
            )

    @tf.function
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    @tf.function
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * .5) + mean

    @tf.function
    def decode(self, z, batch_id, apply_sigmoid=False):
        logits = self.decoder(tf.concat([z], axis=1))
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
        
    def call(self, inputs, apply_sigmoid=False):
        # x = inputs[0]; batch_id = inputs[1]
        x = inputs; batch_id = None
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
            
        return self.decode(z, batch_id, apply_sigmoid=apply_sigmoid)

@tf.function
def reconstruction_loss(x, x_pred, per_sample=False):
    logpx_z = -tf.reduce_sum((x_pred - x)**2, axis=[1,2,3])
    
    if per_sample:
        return -logpx_z
    else:
        return -tf.reduce_mean(logpx_z)

#todo: add options to keep latent and to keep reconstruction
def eval_model(model, x, chunk_size=5000, return_latent=False):
    rloss = []
    Z = []
    for i in range(0, len(x), chunk_size):
        print('.', end='')
        chunk = x[i:i+chunk_size]
        mean, logvar = model.encode(chunk)
        reconst = model.decode(mean, None)
        if return_latent:
            Z.append(mean.numpy())
        rloss.append(reconstruction_loss(chunk, reconst, per_sample=True).numpy())
    
    if return_latent:
        return np.mean(np.concatenate(rloss)), np.concatenate(Z)
    else:
        return np.mean(np.concatenate(rloss))

def train(model, x_train, x_val, on_epoch_end=None, batch_size=256, n_epochs=5,
            batch_group_size=20, on_batch_group_end=None):
    from tensorflow.keras.optimizers.schedules import ExponentialDecay
    lr_schedule = ExponentialDecay(
        1e-4,
        decay_steps=len(x_train) // batch_size,
        decay_rate=0.5,
        staircase=True)

    losses = []; val_losses = []; rlosses = []; val_rlosses = []
    epochids = []; batchids = []
    learning_rates = []
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule, clipvalue=1)

    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1} of {n_epochs}")

        # Iterate over batches
        ix = np.argsort(np.random.randn(len(x_train)))
        for n, i in enumerate(range(0, len(x_train), batch_size)):
            x_batch = x_train[ix[i:i+batch_size]]
            y_batch = x_train[ix[i:i+batch_size]]

            # Forward pass
            with tf.GradientTape() as tape:
                mean, logvar = model.encode(x_batch)
                z = model.reparameterize(mean, logvar)
                kl_loss = -1 * 0.5 * tf.reduce_mean(
                        tf.reduce_sum(
                            1 + logvar - tf.square(mean) - tf.exp(logvar),
                            axis=1
                        )
                    )
                    
                predictions = model.decode(z, None)
                rloss = reconstruction_loss(y_batch, predictions)
                loss = kl_loss + rloss

            # Backward pass
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Save info
            losses.append(loss.numpy())
            rlosses.append(rloss.numpy())
            batchids.append(n)
            epochids.append(epoch)
            val_losses.append(np.NaN)
            val_rlosses.append(np.NaN)
            learning_rates.append(optimizer.get_config()['learning_rate'])

            if n % batch_group_size == 0:
                print(f'[{int(rloss.numpy())} , {int(kl_loss.numpy())}]', end=' ')
                on_batch_group_end(model, epoch+1, n) if on_batch_group_end is not None else None
        
        print('end of epoch')
        val_rlosses[-1] = eval_model(model, x_val)
        training_meta = pd.DataFrame({'epoch':epochids, 'batch':batchids, 'loss':losses,
            'rloss':rlosses, 'val_rloss':val_rlosses, 'learning_rate':learning_rates})
        print(f'est. val rloss = {val_rlosses[-1]}')

        on_epoch_end(model, epoch+1, training_meta) if on_epoch_end is not None else None

    return training_meta