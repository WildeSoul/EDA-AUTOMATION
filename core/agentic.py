"""
Agentic Pipeline: Plan → Execute → Synthesize
Uses Nemotron Super (planner) + Nemotron Nano (executor) via Ollama.
"""
import re
import textwrap
from core.llm import query_ollama
from config.settings import NEMOTRON_MODELS, CLOUD_MODELS, TIMEOUT_CLOUD, TIMEOUT_LOCAL


def agentic_plan(question: str, data_context: str) -> tuple[str, list[str]]:
    """
    STEP 1 — Super decomposes the question into analytical sub-tasks.
    Returns (raw_plan_text, list_of_steps).
    """
    plan_prompt = textwrap.dedent(f"""\
    You are a principal data scientist designing an analysis plan. Decompose this
    question into exactly 3-5 numbered sub-tasks that cover the FULL analytical depth.

    Each sub-task MUST:
    - NAME specific columns, statistical methods, and expected output format
    - Be INDEPENDENTLY executable with the provided data
    - Include one of: descriptive stats, distribution analysis, correlation check,
      hypothesis test, segmentation, trend analysis, or anomaly detection
    - State the STATISTICAL METHOD (e.g., "compute Pearson r between X and Y",
      "test normality of Z with Shapiro-Wilk", "segment by Q and compare medians")

    ENSURE coverage of:
    - At least 1 sub-task on data quality / distribution shape
    - At least 1 sub-task on relationships / correlations
    - At least 1 sub-task on actionable segmentation or comparison

    ## USER QUESTION
    {question}

    ## AVAILABLE DATA CONTEXT
    {data_context[:4000]}

    ## OUTPUT FORMAT
    Return ONLY the numbered list, one sub-task per line:
    1. [sub-task with specific columns and statistical method]
    2. [sub-task with specific columns and statistical method]
    ...
    """)

    plan_system = (
        "You are a principal data scientist who designs rigorous analysis plans. "
        "Every sub-task must name specific columns, statistical tests, and metrics. "
        "You ensure completeness: data quality, distributions, relationships, and actionable insights."
    )

    planner_model = NEMOTRON_MODELS["planner"]
    raw_plan = query_ollama(planner_model, plan_prompt, system=plan_system,
                            timeout=TIMEOUT_CLOUD)

    # Parse numbered steps
    steps = []
    for line in raw_plan.split("\n"):
        line = line.strip()
        if line and len(line) > 3 and line[0].isdigit() and ("." in line[:3] or ")" in line[:3]):
            step_text = re.sub(r"^\d+[\.\\)]\s*", "", line).strip()
            if step_text:
                steps.append(step_text)

    if not steps:
        steps = [question]

    return raw_plan, steps


def agentic_execute(steps: list[str], data_context: str) -> list[dict]:
    """
    STEP 2 — Nano executes each sub-task against the data context.
    Returns list of {step, result} dicts.
    """
    executor_model = NEMOTRON_MODELS["executor"]
    executor_is_cloud = executor_model in CLOUD_MODELS
    exec_system = (
        "You are a precise quantitative analyst executing a specific analytical task. "
        "RULES:\n"
        "1. Use ONLY the data provided — never fabricate numbers.\n"
        "2. Cite exact column names, values, counts, percentages, p-values.\n"
        "3. For statistical tests, report: test_statistic, p-value, effect_size, interpretation.\n"
        "4. For distributions, report: mean, median, std, skewness, range.\n"
        "5. For correlations, report: coefficient, direction, strength (weak/moderate/strong).\n"
        "6. Keep response focused — answer ONLY this specific sub-task.\n"
        "7. End with one sentence on confidence level and any data quality caveats."
    )

    results = []
    for step in steps:
        exec_prompt = textwrap.dedent(f"""\
        ## TASK
        {step}

        ## DATA
        {data_context[:4000]}

        Answer with EXACT numbers from the data above. Structure your response:
        - **Finding**: The specific answer with numbers
        - **Evidence**: Supporting statistics (means, medians, p-values, counts)
        - **Caveat**: Any data quality issue affecting this finding
        """)

        exec_timeout = TIMEOUT_CLOUD if executor_is_cloud else TIMEOUT_LOCAL
        result = query_ollama(executor_model, exec_prompt, system=exec_system,
                              timeout=exec_timeout)
        results.append({"step": step, "result": result})

    return results


def agentic_synthesize(question: str, plan: str, exec_results: list[dict]) -> str:
    """
    STEP 3 — Super merges the plan + execution results into a polished answer.
    """
    results_text = ""
    for i, r in enumerate(exec_results, 1):
        results_text += f"\n### Sub-task {i}: {r['step']}\n{r['result']}\n"

    synth_prompt = textwrap.dedent(f"""\
    You are a principal data scientist creating an executive-level synthesis.

    ## ORIGINAL QUESTION
    {question}

    ## ANALYSIS PLAN
    {plan}

    ## EXECUTION RESULTS FROM ANALYST
    {results_text}

    ## YOUR TASK
    Synthesize ALL sub-task results into a rigorous, comprehensive answer.

    ### 📌 DIRECT ANSWER
    Lead with the single most important finding. Include the key number.

    ### 📊 STATISTICAL EVIDENCE
    Summarize the hard numbers: means, medians, correlations, p-values,
    effect sizes from the execution results. Use a table if comparing groups.

    ### 🔍 KEY INSIGHTS (3-5)
    Numbered insights — each must cite a SPECIFIC number from the execution results.
    For each insight, explain the MECHANISM (why this pattern exists).

    ### ⚠️ DATA QUALITY & CAVEATS
    Flag sample size issues, missing data impact, violated assumptions,
    or any reason the findings might not generalize.

    ### ✅ ACTIONABLE RECOMMENDATIONS
    Numbered steps with specific thresholds, filters, or code snippets.
    Each must be directly tied to an insight above.

    ### 🎯 CONFIDENCE LEVEL
    Rate High/Medium/Low.
    - State key ASSUMPTIONS made
    - State what EVIDENCE would change the conclusion
    - State recommended SAMPLE SIZE for stronger conclusions

    CRITICAL: Use EXACT numbers from execution results. NEVER fabricate data.
    Cross-check that insights are consistent across sub-tasks.
    """)

    synth_system = (
        "You are a principal data scientist creating executive-level synthesis of "
        "analytical findings. You combine statistical rigor with business clarity. "
        "Cite exact numbers, quantify uncertainty, and provide specific actionable steps."
    )

    planner_model = NEMOTRON_MODELS["planner"]
    return query_ollama(planner_model, synth_prompt, system=synth_system,
                        timeout=TIMEOUT_CLOUD)
