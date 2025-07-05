from typing import Any, Callable, List


class TransformPipeline:
    """
    A simple class to compose and apply a sequence of transformations.

    This class allows for creating a pipeline of callable transformations that
    are applied sequentially to an input data object.
    """

    def __init__(self, transforms: List[Callable]):
        """
        Initializes the TransformPipeline.

        Args:
            transforms: A list of callable objects (functions or classes with
                        __call__) that will form the pipeline.
        """
        self.transforms = transforms

    def __call__(self, data: Any) -> Any:
        """
        Applies the sequence of transformations to the data.

        Args:
            data: The initial data to be transformed.

        Returns:
            The transformed data after applying all callables in the pipeline.
        """
        for transform in self.transforms:
            data = transform(data)
        return data
