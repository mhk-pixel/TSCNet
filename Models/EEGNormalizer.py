import numpy as np
class EEGNormalizer:
    """
    EEG Normalization
    Supported methods:
        - none
        - zscore
        - zscore_robust
        - winsor_zscore
    Supported levels:
        - per_channel
        - per_sample
    """
    EPS = 1e-8

    def __init__(self, method, level, clip_sigma):
        self.method = method
        self.level = level
        self.clip_sigma = clip_sigma
        self.stats = None
        
    def normalize(self, train_data, val_data=None, test_data=None):

        if self.method == 'none':
            return train_data, val_data, test_data
        ##################################################################
        
        print(f"\nApplying {self.method} normalization ({self.level})...")
        if self.level == 'per_channel':
            self.fit(train_data)
            train_data = self.transform(train_data)
            
            if val_data is not None:
                val_data = self.transform(val_data)

            if test_data is not None:
                
               
                test_data = self.transform(test_data)
        ##################################################################
        
        elif self.level == 'per_sample':

            train_data = self._normalize_per_sample(train_data)
            if val_data is not None:
                val_data = self._normalize_per_sample(val_data)
            if test_data is not None:
                test_data = self._normalize_per_sample(test_data)
        else:
            raise ValueError(f"Unknown normalization level: {self.level}")

        return train_data, val_data, test_data
    #########################################################################
    def fit(self, X):

        assert X.ndim == 3

        N, C, T = X.shape

        flat = X.transpose(0, 2, 1).reshape(-1, C)

        self.stats = {}
        for c in range(C):
            self.stats[c] = self._compute_stats(flat[:, c])
        return self
    ###########################################################################
    def transform(self, X):

        if self.stats is None:
            raise RuntimeError("fit() must be called before transform().")
        N, C, T = X.shape
        flat = X.transpose(0, 2, 1).reshape(-1, C)

        for c in range(C):
            flat[:, c] = self._apply_stats(
                flat[:, c],
                self.stats[c]
            )

        return flat.reshape(N, T, C).transpose(0, 2, 1)
    #############################################################################
    def _compute_stats(self, x):

        if self.method == 'zscore':
            center = np.mean(x)
            scale = np.std(x)
            return {
                'center': center,
                'scale': max(scale, self.EPS)
            }
            
        elif self.method == 'zscore_robust':
            center = np.median(x)
            mad = np.median(np.abs(x - center))
            scale = 1.4826 * mad
            return {
                'center': center,
                'scale': max(scale, self.EPS)
            }

        elif self.method == 'winsor_zscore':
            q_low, q_high = np.quantile(x, [0.01, 0.99])
            x_clip = np.clip(x, q_low, q_high)
            center = np.mean(x_clip)
            scale = np.std(x_clip)
            return {
                'center': center,
                'scale': max(scale, self.EPS),
                'q_low': q_low,
                'q_high': q_high
            }
        elif self.method == 'minmax':
            center = np.min(x)
            scale = np.max(x) - center
            return {
                'center': center,
                'scale': max(scale, self.EPS)
            }
        raise ValueError(f"Unsupported method: {self.method}")

    def _apply_stats(self, x, stats):

        if self.method == 'winsor_zscore':
            x = np.clip(
                x,
                stats['q_low'],
                stats['q_high']
            )
        x = (x - stats['center']) / stats['scale']
        return np.clip(
            x,
            -self.clip_sigma,
            self.clip_sigma
        )
    ###############################################################################axis=1, keepdims=True
    def _normalize_per_sample(self, X):

        out = np.empty_like(X)
    
        for i in range(X.shape[0]):
    
            sample = X[i]
    
            if self.method == 'minmax':
    
                sample_min = sample.min(axis=1, keepdims=True)
                sample_max = sample.max(axis=1, keepdims=True)
                sample = (sample - sample_min) / (sample_max - sample_min + self.EPS)

                out[i] = sample
                
    
            elif self.method == 'zscore':
    
                center = sample.mean(axis=1, keepdims=True)
                scale = sample.std(axis=1, keepdims=True)
    
            elif self.method == 'zscore_robust':
    
                center = np.median(sample, axis=1, keepdims=True)
                mad = np.median(
                    np.abs(sample - center),
                    axis=1,
                    keepdims=True
                )
                scale = 1.4826 * mad
    
            elif self.method == 'winsor_zscore':
    
                q_low = np.quantile(
                    sample,
                    0.01,
                    axis=1,
                    keepdims=True
                )
                q_high = np.quantile(
                    sample,
                    0.99,
                    axis=1,
                    keepdims=True
                )
    
                sample = np.clip(
                    sample,
                    q_low,
                    q_high
                )
    
                center = sample.mean(axis=1, keepdims=True)
                scale = sample.std(axis=1, keepdims=True)
    
            else:
                raise ValueError(
                    f"Unsupported method: {self.method}"
                )
    
           
    
        return out