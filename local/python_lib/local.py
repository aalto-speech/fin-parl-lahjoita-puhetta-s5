import torch
from speechbrain.dataio.batch import (default_convert, mod_default_collate, recursive_to, recursive_pin_memory)
from speechbrain.lobes.models.transformer.Transformer import ( 
        TransformerInterface,
)
import speechbrain.utils.checkpoints as checkpoints
import math

class NoSchedule:
    def __init__(self, lr, **kwargs):
        self.lr = lr

    def __call__(self, *args, **kwargs):
        return self.lr, self.lr

@checkpoints.register_checkpoint_hooks
class NewBobSchedulerWithWarmup:
    """Scheduler with new-bob technique, used for LR annealing.
    The learning rate is annealed based on the validation performance.
    In particular: if (past_loss-current_loss)/past_loss< impr_threshold:
    lr=lr * annealing_factor.
    Arguments
    ---------
    initial_value : float
        The initial hyperparameter value.
    annealing_factor : float
        It is annealing factor used in new_bob strategy.
    improvement_threshold : float
        It is the improvement rate between losses used to perform learning
        annealing in new_bob strategy.
    patient : int
        When the annealing condition is violated patient times,
        the learning rate is finally reduced.
    Example
    -------
    >>> scheduler = NewBobScheduler(initial_value=1.0)
    >>> scheduler(metric_value=10.0)
    (1.0, 1.0)
    >>> scheduler(metric_value=2.0)
    (1.0, 1.0)
    >>> scheduler(metric_value=2.5)
    (1.0, 0.5)
    """

    def __init__(
        self,
        initial_value,
        highest_value,
        warmup_epochs=10,
        annealing_factor=0.5,
        improvement_threshold=0.0025,
        patient=0,
    ):
        self.hyperparam_value = initial_value
        self.annealing_factor = annealing_factor
        self.improvement_threshold = improvement_threshold
        self.patient = patient
        self.metric_values = []
        self.current_patient = self.patient
        self.curr_epoch = 1
        if warmup_epochs <= 1:
            raise ValueError("Warmup epochs needs to be > 1")
        self.warmup_epochs = warmup_epochs
        self.interpolation_constant = math.pow((highest_value / initial_value), 1/(self.warmup_epochs-1))

    def __call__(self, metric_value):
        """Returns the current and new value for the hyperparameter.
        Arguments
        ---------
        metric_value : int
            A number for determining whether to change the hyperparameter value.
        """
        old_value = new_value = self.hyperparam_value
        next_epoch = self.curr_epoch + 1

        if next_epoch <= self.warmup_epochs:
            new_value = self.interpolation_constant * old_value
        elif len(self.metric_values) > 0:
            prev_metric = self.metric_values[-1]
            # Update value if improvement too small and patience is 0
            if prev_metric == 0:  # Prevent division by zero
                improvement = 0
            else:
                improvement = (prev_metric - metric_value) / prev_metric
            if improvement < self.improvement_threshold:
                if self.current_patient == 0:
                    new_value *= self.annealing_factor
                    self.current_patient = self.patient
                else:
                    self.current_patient -= 1

        # Store relevant info
        self.metric_values.append(metric_value)
        self.hyperparam_value = new_value
        self.curr_epoch = next_epoch
        return old_value, new_value

    @checkpoints.mark_as_saver
    def save(self, path):
        """Saves the current metrics on the specified path."""
        data = {
            "hyperparam_value": self.hyperparam_value,
            "metric_values": self.metric_values,
            "current_patient": self.current_patient,
            "curr_epoch": self.curr_epoch,
        }
        torch.save(data, path)

    @checkpoints.mark_as_loader
    def load(self, path, end_of_epoch=False, device=None):
        """Loads the needed information."""
        del end_of_epoch  # Unused in this class
        del device  # Unused in here
        data = torch.load(path)
        self.hyperparam_value = data["hyperparam_value"]
        self.metric_values = data["metric_values"]
        self.current_patient = data["current_patient"]
        self.curr_epoch = data["curr_epoch"]


class Batch:
    def __init__(
        self,
        examples,
        device_prep_keys=None,
        apply_default_convert=True,
    ):
        self.__length = len(examples)
        self.__keys = list(examples[0].keys())
        self.__device_prep_keys = []
        for key in self.__keys:
            values = [example[key] for example in examples]
            # Default convert usually does the right thing (numpy2torch etc.)
            if apply_default_convert:
                values = default_convert(values)
            values = mod_default_collate(values)
            setattr(self, key, values)
            if (device_prep_keys is not None and key in device_prep_keys) or (
                device_prep_keys is None and isinstance(values[0], torch.Tensor)
            ):
                self.__device_prep_keys.append(key)

    def __len__(self):
        return self.__length

    def __getitem__(self, key):
        if key in self.__keys:
            return getattr(self, key)
        else:
            raise KeyError(f"Batch doesn't have key: {key}")

    def __iter__(self):
        """Iterates over the different elements of the batch.

        Example
        -------
        >>> batch = PaddedBatch([
        ...     {"id": "ex1", "val": torch.Tensor([1.])},
        ...     {"id": "ex2", "val": torch.Tensor([2., 1.])}])
        >>> ids, vals = batch
        >>> ids
        ['ex1', 'ex2']
        """
        return iter((getattr(self, key) for key in self.__keys))

    def pin_memory(self):
        """In-place, moves relevant elements to pinned memory."""
        for key in self.__device_prep_keys:
            value = getattr(self, key)
            pinned = recursive_pin_memory(value)
            setattr(self, key, pinned)
        return self

    def to(self, *args, **kwargs):
        """In-place move/cast relevant elements.

        Passes all arguments to torch.Tensor.to, see its documentation.
        """
        for key in self.__device_prep_keys:
            value = getattr(self, key)
            moved = recursive_to(value, *args, **kwargs)
            setattr(self, key, moved)
        return self

    def at_position(self, pos):
        """Fetch an item by its position in the batch."""
        key = self.__keys[pos]
        return getattr(self, key)

    @property
    def batchsize(self):
        return self.__length


class TransformerAM(TransformerInterface):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, num_decoder_layers=0, **kwargs)

    def forward(self, x, src_key_padding_mask=None):
        if self.causal:
            attn_mask = get_lookahead_mask(x)
        else:
            attn_mask = None
        encoder_output, _ = self.encoder(
            src=x,
            src_mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        return encoder_output
