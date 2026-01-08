**Large Language Model Analysis of European Farmer Protests (2023-2024)**
This repository contains the code and data processing workflows for the study "Large language model analysis reveals key reasons behind massive farmer protests in Europe."

The project utilizes Large Language Models (LLMs) to systematically analyze and categorize the reasons behind 4,642 farmer protests across Europe between November 2023 and March 2024.

**Project Overview**
In late 2023 and early 2024, massive farmer protests erupted across Europe. This project aims to understand the drivers of these protests by applying Natural Language Processing (NLP) techniques to event descriptions.

**Key Objectives:**

- Identify the diverse reasons behind farmer protests using an LLM-human collaborative approach.

- Classify individual protest events into specific categories (e.g., subsidy cuts, opposition to imports, environmental regulations).

- Analyze spatial and temporal trends of these protests across European countries.


**Methodology**
The analysis workflow consists of two main components powered by OpenAI's GPT models:

1. **The Proposer (Category Generation)**
Goal: To discover and define the specific reasons for protests from raw text descriptions.

Approach: An iterative process using GPT-4 to analyze subsets of protest descriptions and propose categories. These were refined by human experts to create a final codebook of 18 protest reasons (e.g., "Rising production costs", "Opposition to EU free-trade agreements").

2. **The Classifier (Multi-label Classification)**
Goal: To assign one or more specific reasons to each protest event.

Approach: A zero-shot multi-label classification model using GPT-4o/GPT-4o-mini. The model analyzes the text description of an event and assigns relevant labels based on the definitions generated in step 1.

Validation: The model's performance was validated against human annotators to ensure accuracy.