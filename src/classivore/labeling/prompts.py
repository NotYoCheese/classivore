#!/usr/bin/env python3
"""Prompt construction for two-stage hierarchical labeling.

Stage 1: Tier-1 triage — identify relevant top-level categories.
Stage 2: Subtree classification — assign specific categories from selected subtrees.

Prompts use XML-delimited sections with enriched category descriptions and boundaries.
Uses category `name` field (not `display_name`) to avoid multi-part name errors.
"""

# --- Few-shot examples for stage 2 ---

FEW_SHOT_EXAMPLES = [
    {
        "text": (
            "The 2026 Honda Civic Sedan continues to refine what made previous "
            "generations popular. Our test drive revealed improved handling, a quieter "
            "cabin, and Honda's latest infotainment system. Starting at $24,500, it "
            "competes directly with the Toyota Corolla and Hyundai Elantra. Fuel economy "
            "ratings of 33 city / 42 highway make it one of the most efficient in its class."
        ),
        "response": {
            "reasoning": (
                "This is a car review focused on a specific sedan model. The primary "
                "topic is the sedan body style with detailed performance and pricing "
                "information, which also touches on auto buying considerations."
            ),
            "categories": [
                {"name": "Sedan", "confidence": 0.95},
                {"name": "Auto Buying and Selling", "confidence": 0.62},
            ],
        },
    },
    {
        "text": (
            "Researchers at MIT have developed a new aerodynamic model for passenger "
            "vehicles that could improve fuel efficiency by 15%. The study, published "
            "in the Journal of Fluid Dynamics, used computational simulations to "
            "optimize body panel shapes. Automakers including Toyota and BMW have "
            "expressed interest in licensing the technology for their 2028 model lines."
        ),
        "response": {
            "reasoning": (
                "This article bridges automotive engineering and science. The primary "
                "focus is on vehicle aerodynamics research (automotive), but the "
                "scientific methodology and academic context make Science equally "
                "relevant. The content spans Auto Body Styles broadly rather than "
                "any specific body type."
            ),
            "categories": [
                {"name": "Auto Body Styles", "confidence": 0.82},
                {"name": "Science", "confidence": 0.78},
            ],
        },
    },
    {
        "text": (
            "Planning a family road trip? Here are 10 essential tips for traveling "
            "with kids by car. From choosing the right vehicle to packing entertainment, "
            "we cover everything you need for a stress-free journey. Don't forget to "
            "check your auto insurance coverage for long-distance travel and consider "
            "roadside assistance membership."
        ),
        "response": {
            "reasoning": (
                "This is primarily family travel content about road trips. It "
                "references automotive topics (vehicle choice, insurance, roadside "
                "assistance) as supporting details. Family Travel is the most specific "
                "match, with Auto Insurance as a secondary topic."
            ),
            "categories": [
                {"name": "Family Travel", "confidence": 0.90},
                {"name": "Auto Insurance", "confidence": 0.55},
            ],
        },
    },
]


def build_stage1_system(tier1_categories):
    """Build the stage 1 system prompt for tier-1 triage.

    Args:
        tier1_categories: List of tier-1 category dicts with name, description, boundaries.

    Returns:
        System prompt string.
    """
    cat_xml = []
    for cat in tier1_categories:
        cat_xml.append(
            f'<category name="{cat["name"]}">\n'
            f'<description>{cat.get("description", "")}</description>\n'
            f'<boundary>{cat.get("boundaries", "")}</boundary>\n'
            f'</category>'
        )

    taxonomy_section = "\n\n".join(cat_xml)

    # Build the valid names list for emphasis
    valid_names = ", ".join(f'"{c["name"]}"' for c in tier1_categories)

    return f"""You are an expert content categorizer. Identify which top-level content categories are relevant to the given web page.

<taxonomy>
Below are the ONLY {len(tier1_categories)} valid top-level content categories. You MUST NOT use any category name that is not in this list.

{taxonomy_section}
</taxonomy>

<instructions>
1. Read the page content carefully.
2. Identify ALL top-level categories relevant to the content, even tangentially.
3. Assign a confidence score from 0.0 to 1.0 for each relevant category.
4. Include any category with confidence >= 0.2. Err on inclusion — missing a relevant category is worse than including an extra one.
5. CRITICAL: You MUST only use category names that appear EXACTLY as listed in the taxonomy above. The valid names are: {valid_names}. Do NOT invent, abbreviate, or paraphrase category names.
6. Respond with ONLY a compact single-line JSON object. No indentation, no newlines within the JSON, no markdown.
</instructions>

<output_format>
{{"categories":[{{"name":"CategoryName","confidence":0.85}}]}}
</output_format>"""


def build_stage2_system(subtrees, max_labels, min_confidence):
    """Build the stage 2 system prompt for subtree classification.

    Args:
        subtrees: Dict of tier1_name → list of category dicts (all depths in subtree).
        max_labels: Maximum number of labels to assign.
        min_confidence: Minimum confidence threshold.

    Returns:
        System prompt string.
    """
    taxonomy_xml = []
    for tier1_name, categories in subtrees.items():
        cat_entries = []
        for cat in categories:
            parent = cat.get("path", [])
            parent_name = parent[-2] if len(parent) >= 2 else ""
            cat_entries.append(
                f'  <category name="{cat["name"]}" depth="{cat["depth"]}" parent="{parent_name}">\n'
                f'    <description>{cat.get("description", "")}</description>\n'
                f'    <boundary>{cat.get("boundaries", "")}</boundary>\n'
                f'  </category>'
            )
        taxonomy_xml.append(
            f'<subtree name="{tier1_name}">\n' +
            "\n".join(cat_entries) +
            f'\n</subtree>'
        )

    taxonomy_section = "\n\n".join(taxonomy_xml)
    examples_section = _format_few_shot_examples()

    return f"""You are an expert content categorizer specializing in hierarchical taxonomy classification.

<task>
Assign the most specific applicable categories from the taxonomy below. Categories at any depth are valid — use a non-leaf category when content spans multiple of its children equally.
</task>

<taxonomy>
{taxonomy_section}
</taxonomy>

<examples>
{examples_section}
</examples>

<instructions>
1. Read the content carefully to understand the primary subject and secondary topics.
2. Think step-by-step, working from general to specific.
3. Assign up to {max_labels} categories. Prefer the most specific applicable category.
4. Use non-leaf categories when content covers multiple subcategories equally.
5. Confidence scores: 0.0 to 1.0. Only include categories above {min_confidence}.
6. Use the EXACT category name as shown in the taxonomy above.
7. Explain your reasoning before listing categories.
</instructions>

<output_format>
Respond with ONLY a JSON object. No markdown fences.
{{
  "reasoning": "2-3 sentence analysis of content and category choices.",
  "categories": [
    {{"name": "ExactCategoryName", "confidence": 0.92}},
    {{"name": "AnotherCategory", "confidence": 0.71}}
  ]
}}
</output_format>"""


def _format_few_shot_examples():
    """Format few-shot examples as XML for inclusion in stage 2 prompt."""
    import json

    parts = []
    for example in FEW_SHOT_EXAMPLES:
        response_json = json.dumps(example["response"], indent=2)
        parts.append(
            f'<example>\n'
            f'<content>\n{example["text"]}\n</content>\n'
            f'<response>\n{response_json}\n</response>\n'
            f'</example>'
        )
    return "\n\n".join(parts)


def build_user_message(page_text, url="", truncation_words=3000):
    """Build the user message containing page content.

    Args:
        page_text: Full extracted page text.
        url: Optional URL for context.
        truncation_words: Max words to include.

    Returns:
        User message string.
    """
    words = page_text.split()
    if len(words) > truncation_words:
        page_text = " ".join(words[:truncation_words])

    parts = []
    if url:
        parts.append(f"URL: {url}")
    parts.append(f"Text:\n{page_text}")

    return "\n\n".join(parts)


def get_subtree_categories(categories, tier1_names):
    """Get all categories belonging to the specified tier-1 subtrees.

    Args:
        categories: Full list of category dicts.
        tier1_names: Set of tier-1 category names to include.

    Returns:
        Dict of tier1_name → list of category dicts (all depths).
    """
    subtrees = {name: [] for name in tier1_names}

    for cat in categories:
        path = cat.get("path", [])
        if not path:
            continue
        tier1 = path[0]
        if tier1 in subtrees:
            subtrees[tier1].append(cat)

    return subtrees
