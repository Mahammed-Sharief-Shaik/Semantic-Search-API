### Semantic Cache Threshold Calibration

During experimentation, cosine similarity values between semantically related queries typically fell between **0.70 and 0.80** using the `all-MiniLM-L6-v2` embedding model.

Examples observed during testing:

* "space launch" vs "NASA shuttle launch" → 0.775
* "space launch" vs "satellite launch" → 0.774

Because these values were below 0.85, using a threshold of 0.85 resulted in very few cache hits.

After empirical evaluation, the threshold was adjusted to **0.75**, which allowed the cache to correctly detect semantically similar queries while still avoiding unrelated matches.
