## SDR: Succinct Document Representation

This package provides code for the paper *SDR: Efficient Neural Re-ranking using Succinct DocumentRepresentation*. 

Paper: https://aclanthology.org/2022.acl-long.457/

Bibtex entry:

```
@inproceedings{cohen-2022-sdr,
	title = "{SDR}: Efficient Neural Re-ranking using Succinct DocumentRepresentation",
	author = "Cohen, Nachshon  and
	Portnoy, Amit  and
	Fetahu, Besnik  and
	Ingber, Amir",
	booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing",
	year = "2022"
}
```

## Steps to run the code

1. Train a late interaction model: `late_interaction_baseline.trainer`
2. Generate embedding vectors: `late_interaction_baseline.generate_embeddings`
3. Train auto-encoder with side information `auto_encoder.ae_modeling_training`. 
     Note: this trains multiple auto encoders and multiple number of features.  
4. Experiment with different auto-encoders and different quantization `experiments.run_experiment`


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
