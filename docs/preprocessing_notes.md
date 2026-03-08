## Dataset Exploration and Preprocessing Decisions

### Initial Dataset Inspection

The provided dataset contains two archives:

* **20_newsgroups.tar.gz** (~17 MB) — full dataset with ~20,000 articles
* **mini_newsgroups.tar.gz** (~1.8 MB) — smaller subset used for development and testing

Each dataset is organized as:

```
newsgroup_name/
    article_1.txt
    article_2.txt
    ...
```

Each folder represents a **topic category**, and each file contains a **single Usenet post**.

During initial inspection of several documents, it became clear that each article contains three main components:

1. **Header Metadata**
2. **Message Body**
3. **Footer / Signature**

Example structure observed:

```
Header metadata
---------------
From:
Subject:
Organization:
Message-ID:
References:
Path:
Date:

Message body
------------
Actual discussion text written by the author.

Footer / Signature
------------------
User signatures, ASCII art, disclaimers, or contact information.
```

---

### Observed Noise in the Dataset

Several types of non-semantic information were consistently present in the dataset:

**1. Email/Usenet Headers**

Examples:

```
Path: cantaloupe.srv.cs.cmu.edu!...
Message-ID: <1993Apr19...>
Organization: IBM Research
References: <previous message IDs>
Distribution: usa
```

These fields describe **message routing and metadata**, not the semantic content of the discussion.

Including them in the text would introduce large amounts of irrelevant tokens into the embedding model.

---

**2. Quoted Reply Chains**

Many messages contain quoted replies from earlier discussions:

```
> In article <...> someone writes:
> Previous discussion text...
```

These quotes often repeat text from other posts and create **duplicate semantic information**, which can distort clustering and embedding similarity.

---

**3. Email Signatures / Footers**

Many articles end with signature blocks such as:

```
--
Rob Strom
IBM Research
Yorktown Heights, NY
```

or decorative ASCII blocks.

These signatures are **not related to the topic of the message**, and multiple unrelated articles may share identical signatures.

Including them would negatively impact clustering quality.

---

### Preprocessing Strategy

Based on the dataset analysis, the following preprocessing decisions were made:

| Component                                            | Decision | Reason                                   |
| ---------------------------------------------------- | -------- | ---------------------------------------- |
| Email headers (Path, Message-ID, Organization, etc.) | Removed  | Non-semantic routing metadata            |
| Quoted replies (`>` lines)                           | Removed  | Duplicate content from previous messages |
| Signatures / footers                                 | Removed  | Personal information unrelated to topic  |
| Message body                                         | Retained | Contains the actual semantic content     |

One exception is the **Subject line**, which is retained because it often contains useful topic information, such as:

```
Subject: Conference on Manned Lunar Exploration
```

Subject lines frequently summarize the topic of the discussion and can improve semantic search performance.

---

### Additional Filtering

To improve the quality of embeddings, documents that are extremely short or contain insufficient textual content are filtered out.

This avoids generating embeddings for messages such as:

```
"Thanks."
"Agreed."
```

which do not provide meaningful semantic information.

---

### Final Preprocessing Pipeline

The final preprocessing pipeline for each article is:

```
Raw Article
     ↓
Remove headers (except Subject)
     ↓
Remove quoted reply lines
     ↓
Remove signatures / footers
     ↓
Clean whitespace and formatting
     ↓
Filter very short documents
     ↓
Cleaned semantic text
```

The resulting cleaned text is then used to generate embeddings for semantic search, clustering, and caching.
