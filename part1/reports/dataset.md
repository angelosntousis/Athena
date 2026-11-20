# Dataset

Chosen dataset: IMDb Classification dataset

Task: Sentiment Classification

Dataset size: 
24904 - train split
24678 - test split

Short description: A dataset of film reviews from IMDb. Each review is labeled as negative (0) or positive (1).

## Labeling strategy:

$\geq 7$ - marked as positive

$\leq 4$ - marked as negative

## Positive examples:

## Negative examples:

## Labels distribution:

Train positive ratio: 0.501

Train negative ratio: 0.499

Test positive ratio: 0.503

Test negative ratio: 0.497

Dataset is balanced, so accuracy is a reasonable metric

## Length distribution

Mean length: 1326.376

std: 1004.11

median length: 979.0

## Duplicates:

No duplicates because each example is a separate review on a film. 

## Wrong labels:

Since the labeling is done by thresholding the rating, there is no annotation errors. The only possible source of errors is reviewer's mistake: rating irrelevant to the review content. 

## Preprocessing

...
- Lowercasing
- ... becomes one token, ....... becomes ...
- !!!!! becomes !!, ????? becomes ??, --- and - becomes ' - '

## Other patterns

9% of reviews contain the rating like "8/10" or "2 out of 10"














## Citation

Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. 2011. [Learning Word Vectors for Sentiment Analysis](https://aclanthology.org/P11-1015). In *Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies*, pages 142–150, Portland, Oregon, USA. Association for Computational Linguistics.

<details>
<summary>BibTeX</summary>

```bibtex
@inproceedings{maas-etal-2011-learning,
  address = {Portland, Oregon, USA},
  author = {Maas, Andrew L. and Daly, Raymond E. and Pham, Peter T. and Huang, Dan and Ng, Andrew Y. and Potts, Christopher},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  editor = {Lin, Dekang and Matsumoto, Yuji and Mihalcea, Rada},
  month = jun,
  pages = {142--150},
  publisher = {Association for Computational Linguistics},
  title = {Learning Word Vectors for Sentiment Analysis},
  url = {https://aclanthology.org/P11-1015},
  year = {2011},
}
```

</details>