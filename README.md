# Multi-Agent Product Data Enrichment System
## Architecture Document v1.0

---

## 1. Executive Summary

This document outlines the architecture for a **Proof of Concept (POC)** demonstrating a multi-agent system for automated product data enrichment. The system identifies SKUs with missing attributes, acquires data from multiple sources, extracts structured information from unstructured documents, validates compliance, and loads enriched data to a data warehouse.

**Key Objective:** Demonstrate intelligent orchestration where LLM-powered agents handle complex reasoning tasks while deterministic agents handle API calls and data transformations efficiently.

---

## 2. Tech Stack

### Core Framework
| Component | Technology | Purpose |
|-----------|------------|---------|
| **Orchestration** | LangGraph | Multi-agent workflow management, state handling, conditional routing |
| **LLM Provider** | OpenAI GPT-4o / Claude 3.5 Sonnet | Extraction from unstructured data, insight generation |
| **Runtime** | Python 3.11+ | Primary language |

### Backend
| Component | Technology | Purpose |
|-----------|------------|---------|
| **API Framework** | FastAPI | REST endpoints, WebSocket for real-time updates |
| **Async Support** | asyncio | Non-blocking agent execution |
| **Data Validation** | Pydantic | Schema enforcement, type safety |

### Frontend
| Component | Technology | Purpose |
|-----------|------------|---------|
| **UI Framework** | Streamlit | Rapid prototyping, real-time updates |
| **Visualization** | Streamlit components + custom CSS | Pipeline visualization, progress indicators |

### Data & Storage
| Component | Technology | Purpose |
|-----------|------------|---------|
| **Mock Database** | SQLite / In-memory dict | Sample product catalog |
| **Mock Data Warehouse** | SQLite (simulating Snowflake) | Enriched data storage |
| **Document Storage** | Local filesystem | Sample PDFs, images |

### Observability
| Component | Technology | Purpose |
|-----------|------------|---------|
| **Logging** | Python logging + Loguru | Structured activity logs |
| **Tracing** | LangSmith (optional) | Agent execution traces |

---

## 3. System Architecture

### 3.1 High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PRESENTATION LAYER                              │
│                                 (Streamlit)                                  │
├──────────┬──────────┬──────────┬───────────────┬──────────────┬─────────────┤
│ Product  │ Pipeline │ Activity │  Enrichment   │    Data      │   Agent     │
│ Selector │  Visual  │   Log    │   Results     │   Quality    │   Status    │
└────┬─────┴────┬─────┴────┬─────┴───────┬───────┴──────┬───────┴──────┬──────┘
     │          │          │             │              │              │
     └──────────┴──────────┴─────────────┴──────────────┴──────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ORCHESTRATION LAYER                             │
│                                 (LangGraph)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         EnrichmentState                              │    │
│  │  {sku_id, missing_attrs, acquired_data, extracted_data, ...}        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│    ┌───────────┬───────────┬───────┴───────┬───────────┬───────────┐        │
│    ▼           ▼           ▼               ▼           ▼           ▼        │
│ ┌──────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────┐  ┌───────┐   │
│ │Trigger│  │Acquisition│  │Extraction│  │Validation│  │Mapping│  │Insight│   │
│ │Agent │  │  Agent   │  │  Agent   │  │  Agent   │  │ Agent │  │ Agent │   │
│ │      │  │          │  │          │  │          │  │       │  │       │   │
│ │[Pure]│  │ [Pure]   │  │  [LLM]   │  │ [Hybrid] │  │[Pure] │  │ [LLM] │   │
│ └──────┘  └──────────┘  └──────────┘  └──────────┘  └───────┘  └───────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                               DATA LAYER                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │   Mock GDSN     │  │    Mock FDA     │  │  Mock Supplier  │              │
│  │     API         │  │      API        │  │   (PDF/Images)  │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐                                   │
│  │  Product DB     │  │   Snowflake     │                                   │
│  │   (SQLite)      │  │   (SQLite)      │                                   │
│  └─────────────────┘  └─────────────────┘                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Agent Pipeline Flow

```
                              START
                                │
                                ▼
                    ┌───────────────────────┐
                    │    TRIGGER AGENT      │
                    │    ────────────────   │
                    │    • Query product DB │
                    │    • Identify missing │
                    │      attributes       │
                    │    • [Pure Python]    │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   ACQUISITION AGENT   │
                    │   ─────────────────   │
                    │   • Fetch from GDSN   │
                    │   • Fetch from FDA    │
                    │   • Fetch supplier    │
                    │     documents         │
                    │   • [Pure Python]     │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   EXTRACTION AGENT    │
                    │   ─────────────────   │
                    │   • Parse PDFs        │
                    │   • Process images    │
                    │   • Structure data    │
                    │   • [LLM-Powered]     │◄─── GPT-4o / Claude
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   VALIDATION AGENT    │
                    │   ─────────────────   │
                    │   • Compliance checks │
                    │   • Data quality      │
                    │   • Cross-reference   │
                    │   • [Hybrid]          │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │    MAPPING AGENT      │
                    │    ────────────────   │
                    │    • Schema transform │
                    │    • Merge with       │
                    │      existing data    │
                    │    • Load to Snowflake│
                    │    • [Pure Python]    │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │    INSIGHT AGENT      │
                    │    ────────────────   │
                    │    • Generate reports │
                    │    • Risk analysis    │
                    │    • Recommendations  │
                    │    • [LLM-Powered]    │◄─── GPT-4o / Claude
                    └───────────┬───────────┘
                                │
                                ▼
                               END
```

---

## 4. State Management

### 4.1 Central State Schema

```python
from typing import TypedDict, Literal, Optional
from datetime import datetime

class AcquiredData(TypedDict):
    source: Literal["gdsn", "fda", "supplier"]
    data_type: Literal["json", "pdf", "image"]
    content: dict | str  # dict for JSON, base64/path for files
    timestamp: str

class ExtractedData(TypedDict):
    ingredients: Optional[list[str]]
    nutrition_info: Optional[dict]
    allergens: Optional[list[str]]
    certifications: Optional[list[str]]
    weight: Optional[str]
    dimensions: Optional[dict]

class ValidationResult(TypedDict):
    is_valid: bool
    compliance_status: dict[str, bool]  # {"fda": True, "allergen_declaration": True}
    quality_score: float  # 0-100
    issues: list[str]
    warnings: list[str]

class EnrichmentState(TypedDict):
    # Input
    sku_id: str
    product_name: str
    original_data: dict
    missing_attributes: list[str]
    
    # Pipeline outputs
    acquired_data: list[AcquiredData]
    extracted_data: ExtractedData
    validation_result: ValidationResult
    final_enriched_data: dict
    insights: dict
    
    # Metadata
    current_agent: str
    agent_history: list[dict]  # For activity log
    started_at: str
    completed_at: Optional[str]
    status: Literal["running", "completed", "failed"]
```

### 4.2 State Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                           STATE EVOLUTION                           │
└─────────────────────────────────────────────────────────────────────┘

INITIAL STATE (User selects product)
{
    sku_id: "SKU-001",
    product_name: "Organic Peanut Butter",
    original_data: {brand: "NutriGood", weight: "500g"},
    missing_attributes: [],  // To be filled by Trigger
    ...
}
        │
        ▼ [Trigger Agent]
{
    ...
    missing_attributes: ["ingredients", "allergens", "nutrition_info"],
    agent_history: [{agent: "trigger", action: "identified 3 missing attrs", ts: "..."}]
}
        │
        ▼ [Acquisition Agent]
{
    ...
    acquired_data: [
        {source: "gdsn", data_type: "json", content: {...}},
        {source: "supplier", data_type: "pdf", content: "base64..."}
    ],
    agent_history: [..., {agent: "acquisition", action: "fetched from 2 sources"}]
}
        │
        ▼ [Extraction Agent]
{
    ...
    extracted_data: {
        ingredients: ["peanuts", "salt"],
        allergens: ["peanuts"],
        nutrition_info: {calories: 190, fat: 16}
    }
}
        │
        ▼ [Validation Agent]
{
    ...
    validation_result: {
        is_valid: true,
        compliance_status: {fda: true, allergen_declaration: true},
        quality_score: 92.5,
        issues: [],
        warnings: ["High sodium content"]
    }
}
        │
        ▼ [Mapping Agent]
{
    ...
    final_enriched_data: {
        sku_id: "SKU-001",
        brand: "NutriGood",
        weight: "500g",
        ingredients: ["peanuts", "salt"],  // FILLED
        allergens: ["peanuts"],             // FILLED
        nutrition_info: {...}               // FILLED
    }
}
        │
        ▼ [Insight Agent]
{
    ...
    insights: {
        summary: "Successfully enriched 3 attributes...",
        compliance_report: "Product meets FDA requirements...",
        risk_flags: ["Contains major allergen (peanuts)"],
        recommendations: ["Add tree nut cross-contamination warning"]
    },
    status: "completed"
}
```

---

## 5. Agent Specifications

### 5.1 Trigger Agent

```
┌─────────────────────────────────────────────────────────────────────┐
│                          TRIGGER AGENT                               │
├─────────────────────────────────────────────────────────────────────┤
│  Type:        Pure Python (No LLM)                                  │
│  Input:       sku_id, product_name                                  │
│  Output:      missing_attributes, original_data                     │
│  Latency:     < 100ms                                               │
├─────────────────────────────────────────────────────────────────────┤
│  Logic:                                                             │
│  1. Query product database for SKU                                  │
│  2. Compare against REQUIRED_ATTRIBUTES schema                      │
│  3. Identify NULL or empty fields                                   │
│  4. Return list of missing attribute names                          │
├─────────────────────────────────────────────────────────────────────┤
│  Required Attributes Schema:                                        │
│  - product_name (string)                                            │
│  - brand (string)                                                   │
│  - ingredients (list[string])                                       │
│  - nutrition_info (dict)                                            │
│  - allergens (list[string])                                         │
│  - weight (string)                                                  │
│  - certifications (list[string])                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Acquisition Agent

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ACQUISITION AGENT                             │
├─────────────────────────────────────────────────────────────────────┤
│  Type:        Pure Python (No LLM)                                  │
│  Input:       sku_id, missing_attributes                            │
│  Output:      acquired_data (list of sources)                       │
│  Latency:     ~2s (simulated API calls)                             │
├─────────────────────────────────────────────────────────────────────┤
│  Logic:                                                             │
│  1. Attempt GDSN API (primary source, structured JSON)              │
│  2. Check what's still missing after GDSN                           │
│  3. If gaps remain, fetch FDA data (for compliance attributes)      │
│  4. If gaps remain, fetch supplier documents (PDFs/images)          │
│  5. Return all acquired data with source metadata                   │
├─────────────────────────────────────────────────────────────────────┤
│  Source Priority:                                                   │
│  1. GDSN (trust: 95%) - structured, authoritative                   │
│  2. FDA (trust: 90%) - regulatory, compliance-focused               │
│  3. Supplier (trust: 70%) - may be PDFs, less structured            │
├─────────────────────────────────────────────────────────────────────┤
│  Mock Implementation:                                               │
│  - GDSN: Return pre-defined JSON for sample SKUs                    │
│  - FDA: Return compliance status dict                               │
│  - Supplier: Return path to sample PDF/image files                  │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.3 Extraction Agent

```
┌─────────────────────────────────────────────────────────────────────┐
│                        EXTRACTION AGENT                              │
├─────────────────────────────────────────────────────────────────────┤
│  Type:        LLM-Powered                                           │
│  Input:       acquired_data (may include PDFs, images)              │
│  Output:      extracted_data (structured dict)                      │
│  Latency:     ~2-4s (LLM inference)                                 │
├─────────────────────────────────────────────────────────────────────┤
│  Logic:                                                             │
│  1. For each acquired data source:                                  │
│     - If JSON: Parse directly (no LLM needed)                       │
│     - If PDF: Use LLM with document parsing                         │
│     - If Image: Use LLM with vision capability                      │
│  2. Merge extracted attributes from all sources                     │
│  3. Handle conflicts (prefer higher-trust sources)                  │
│  4. Return unified extracted_data dict                              │
├─────────────────────────────────────────────────────────────────────┤
│  LLM Prompt Strategy:                                               │
│  - System: "Extract product attributes from this document..."       │
│  - Include target schema for structured output                      │
│  - Use JSON mode for reliable parsing                               │
├─────────────────────────────────────────────────────────────────────┤
│  Conditional LLM Usage:                                             │
│  - SKIP LLM if all acquired data is already structured JSON         │
│  - USE LLM only for unstructured sources (PDF, image, HTML)         │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.4 Validation Agent

```
┌─────────────────────────────────────────────────────────────────────┐
│                        VALIDATION AGENT                              │
├─────────────────────────────────────────────────────────────────────┤
│  Type:        Hybrid (Rules + Optional LLM)                         │
│  Input:       extracted_data, missing_attributes                    │
│  Output:      validation_result                                     │
│  Latency:     ~1-2s                                                 │
├─────────────────────────────────────────────────────────────────────┤
│  Rule-Based Checks (Pure Python):                                   │
│  1. Completeness: Are all missing_attributes now filled?            │
│  2. Format validation: Nutrition values are numbers, etc.           │
│  3. FDA ingredient check: All ingredients in approved list?         │
│  4. Allergen declaration: If contains allergen, is it declared?     │
│  5. Certification validity: Known certification names?              │
├─────────────────────────────────────────────────────────────────────┤
│  LLM-Assisted Checks (Optional):                                    │
│  - Semantic validation: "Does ingredient list make sense?"          │
│  - Conflict detection: "Nutrition values seem inconsistent"         │
│  - Only triggered if rule-based checks pass but confidence is low   │
├─────────────────────────────────────────────────────────────────────┤
│  Quality Score Calculation:                                         │
│  score = (completeness * 0.4) +                                     │
│          (source_trust * 0.3) +                                     │
│          (validation_pass_rate * 0.3)                               │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.5 Mapping Agent

```
┌─────────────────────────────────────────────────────────────────────┐
│                          MAPPING AGENT                               │
├─────────────────────────────────────────────────────────────────────┤
│  Type:        Pure Python (No LLM)                                  │
│  Input:       extracted_data, original_data, validation_result      │
│  Output:      final_enriched_data, snowflake_write_status           │
│  Latency:     ~1s                                                   │
├─────────────────────────────────────────────────────────────────────┤
│  Logic:                                                             │
│  1. Start with original_data                                        │
│  2. For each missing attribute:                                     │
│     - If extracted_data has value AND validation passed: merge      │
│     - If validation failed: skip (don't corrupt data)               │
│  3. Transform to Snowflake schema (column name mapping)             │
│  4. Write to Snowflake (mock: SQLite insert)                        │
│  5. Return final merged record                                      │
├─────────────────────────────────────────────────────────────────────┤
│  Schema Transformation Example:                                     │
│  extracted_data          →    snowflake_schema                      │
│  ─────────────────────────────────────────────────                  │
│  ingredients             →    INGREDIENT_LIST                       │
│  nutrition_info.calories →    CALORIES_PER_SERVING                  │
│  allergens               →    ALLERGEN_DECLARATIONS                 │
├─────────────────────────────────────────────────────────────────────┤
│  Merge Strategy:                                                    │
│  - ONLY fill NULL fields (never overwrite existing data)            │
│  - Log what was filled for audit trail                              │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.6 Insight Agent

```
┌─────────────────────────────────────────────────────────────────────┐
│                          INSIGHT AGENT                               │
├─────────────────────────────────────────────────────────────────────┤
│  Type:        LLM-Powered                                           │
│  Input:       final_enriched_data, validation_result                │
│  Output:      insights (reports, recommendations)                   │
│  Latency:     ~2-3s                                                 │
├─────────────────────────────────────────────────────────────────────┤
│  Generated Outputs:                                                 │
│  1. Enrichment Summary                                              │
│     - What was missing, what was filled, from which sources         │
│  2. Compliance Report                                               │
│     - FDA status, labeling requirements met/unmet                   │
│  3. Nutrition Analysis                                              │
│     - Health insights (high sugar, low protein, etc.)               │
│  4. Risk Flags                                                      │
│     - Allergen risks, regulatory concerns                           │
│  5. Recommendations                                                 │
│     - Suggested label changes, additional certifications            │
├─────────────────────────────────────────────────────────────────────┤
│  LLM Prompt:                                                        │
│  "Analyze this enriched product data and generate:                  │
│   1. A brief summary of the enrichment process                      │
│   2. Compliance status with FDA regulations                         │
│   3. Any health/nutrition concerns                                  │
│   4. Risk flags for retailers                                       │
│   5. Actionable recommendations                                     │
│   Product data: {final_enriched_data}                               │
│   Validation results: {validation_result}"                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. LangGraph Implementation

### 6.1 Graph Definition

```python
from langgraph.graph import StateGraph, END

# Define the graph
workflow = StateGraph(EnrichmentState)

# Add nodes (agents)
workflow.add_node("trigger", trigger_agent)
workflow.add_node("acquisition", acquisition_agent)
workflow.add_node("extraction", extraction_agent)
workflow.add_node("validation", validation_agent)
workflow.add_node("mapping", mapping_agent)
workflow.add_node("insight", insight_agent)

# Define edges (flow)
workflow.set_entry_point("trigger")
workflow.add_edge("trigger", "acquisition")
workflow.add_edge("acquisition", "extraction")
workflow.add_edge("extraction", "validation")
workflow.add_edge("validation", "mapping")
workflow.add_edge("mapping", "insight")
workflow.add_edge("insight", END)

# Compile
app = workflow.compile()
```

### 6.2 Graph Visualization

```
                    ┌─────────────────────────────────────┐
                    │           LangGraph Flow            │
                    └─────────────────────────────────────┘

                              ┌─────────┐
                              │  START  │
                              └────┬────┘
                                   │
                                   ▼
                            ┌──────────────┐
                            │   trigger    │
                            └──────┬───────┘
                                   │
                                   ▼
                            ┌──────────────┐
                            │ acquisition  │
                            └──────┬───────┘
                                   │
                                   ▼
                            ┌──────────────┐
                            │  extraction  │
                            └──────┬───────┘
                                   │
                                   ▼
                            ┌──────────────┐
                            │  validation  │
                            └──────┬───────┘
                                   │
                                   ▼
                            ┌──────────────┐
                            │   mapping    │
                            └──────┬───────┘
                                   │
                                   ▼
                            ┌──────────────┐
                            │   insight    │
                            └──────┬───────┘
                                   │
                                   ▼
                              ┌─────────┐
                              │   END   │
                              └─────────┘
```

### 6.3 Conditional Routing (Advanced)

For production, you might want conditional routing:

```python
def route_after_acquisition(state: EnrichmentState) -> str:
    """Decide if extraction is needed based on data types."""
    acquired = state["acquired_data"]
    
    has_unstructured = any(
        d["data_type"] in ["pdf", "image"] for d in acquired
    )
    
    if has_unstructured:
        return "extraction"  # Need LLM to parse
    else:
        return "validation"  # Skip extraction, data already structured

# Add conditional edge
workflow.add_conditional_edges(
    "acquisition",
    route_after_acquisition,
    {
        "extraction": "extraction",
        "validation": "validation"
    }
)
```

---

## 7. UI Architecture (Streamlit)

### 7.1 Layout Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PRODUCT DATA ENRICHMENT SYSTEM                          │
│                          Multi-Agent POC Demo                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─── SIDEBAR ────────────────────┐  ┌─── MAIN AREA ─────────────────────┐ │
│  │                                │  │                                    │ │
│  │  📦 SELECT PRODUCT             │  │  PIPELINE VISUALIZATION            │ │
│  │  ┌──────────────────────────┐  │  │  ┌────┐ ┌────┐ ┌────┐ ┌────┐     │ │
│  │  │ ▼ SKU-001: Peanut Butter │  │  │  │ T  │→│ A  │→│ E  │→│ V  │→... │ │
│  │  │   SKU-002: Organic Milk  │  │  │  │ ✓  │ │ ⟳  │ │ ○  │ │ ○  │     │ │
│  │  │   SKU-003: Vitamin D     │  │  │  └────┘ └────┘ └────┘ └────┘     │ │
│  │  └──────────────────────────┘  │  │                                    │ │
│  │                                │  ├────────────────────────────────────┤ │
│  │  [▶ Start Enrichment]          │  │                                    │ │
│  │                                │  │  ACTIVITY LOG                      │ │
│  │  ─────────────────────────     │  │  ┌────────────────────────────┐   │ │
│  │                                │  │  │ [10:23:01] Trigger: Start  │   │ │
│  │  📊 CURRENT STATUS             │  │  │ [10:23:01] Trigger: Found  │   │ │
│  │  ┌──────────────────────────┐  │  │  │            3 missing attrs │   │ │
│  │  │ Agent: Acquisition       │  │  │  │ [10:23:02] Acquisition:    │   │ │
│  │  │ Status: Running ⟳        │  │  │  │            Fetching GDSN   │   │ │
│  │  │ Time: 1.2s               │  │  │  │ ...                        │   │ │
│  │  └──────────────────────────┘  │  │  └────────────────────────────┘   │ │
│  │                                │  │                                    │ │
│  │  ─────────────────────────     │  ├────────────────────────────────────┤ │
│  │                                │  │                                    │ │
│  │  📈 DATA QUALITY SCORE         │  │  ENRICHMENT RESULTS                │ │
│  │  ┌──────────────────────────┐  │  │                                    │ │
│  │  │ ████████████░░░░  78%    │  │  │  📥 Acquired Data                  │ │
│  │  │                          │  │  │     Source: GDSN + Supplier PDF   │ │
│  │  │ Completeness: 90%        │  │  │                                    │ │
│  │  │ Source Trust: 85%        │  │  │  📄 Extracted Data                 │ │
│  │  │ Validation:   70%        │  │  │     ingredients: [peanuts, salt]  │ │
│  │  └──────────────────────────┘  │  │     allergens: [peanuts]          │ │
│  │                                │  │                                    │ │
│  └────────────────────────────────┘  │  ✓ Validation                      │ │
│                                      │     FDA: PASSED                    │ │
│                                      │     Allergens: PASSED              │ │
│                                      │                                    │ │
│                                      │  💡 Insights                       │ │
│                                      │     "Product meets compliance..."  │ │
│                                      │                                    │ │
│                                      └────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Component Breakdown

| Component | Streamlit Element | Update Mechanism |
|-----------|-------------------|------------------|
| Product Selector | `st.selectbox` | Static |
| Start Button | `st.button` | Click triggers pipeline |
| Pipeline Visual | `st.columns` + custom HTML/CSS | `st.session_state` update |
| Activity Log | `st.container` + `st.write` | Append on each agent step |
| Quality Score | `st.progress` + `st.metric` | Update after validation |
| Enrichment Results | `st.expander` + `st.json` | Update after each agent |

### 7.3 Real-Time Updates Strategy

```python
# Using Streamlit's session state + rerun pattern

import streamlit as st
import time

def run_pipeline(sku_id: str):
    """Run pipeline with real-time UI updates."""
    
    # Initialize state
    if "pipeline_status" not in st.session_state:
        st.session_state.pipeline_status = {}
        st.session_state.logs = []
    
    # Create placeholder for dynamic content
    status_placeholder = st.empty()
    log_placeholder = st.empty()
    
    agents = ["trigger", "acquisition", "extraction", "validation", "mapping", "insight"]
    
    for agent in agents:
        # Update status
        st.session_state.pipeline_status[agent] = "running"
        status_placeholder.write(render_pipeline_visual())
        
        # Run agent (this is where actual agent logic goes)
        result = run_agent(agent, state)
        
        # Log
        st.session_state.logs.append(f"[{timestamp()}] {agent}: completed")
        log_placeholder.write(render_logs())
        
        # Mark complete
        st.session_state.pipeline_status[agent] = "completed"
        status_placeholder.write(render_pipeline_visual())
```

---

## 8. Mock Data Strategy

### 8.1 Sample Products

```python
SAMPLE_PRODUCTS = [
    {
        "sku_id": "SKU-001",
        "product_name": "Organic Peanut Butter",
        "brand": "NutriGood",
        "existing_data": {
            "brand": "NutriGood",
            "weight": "500g"
        },
        "expected_missing": ["ingredients", "allergens", "nutrition_info", "certifications"]
    },
    {
        "sku_id": "SKU-002",
        "product_name": "Whole Milk 1L",
        "brand": "FarmFresh",
        "existing_data": {
            "brand": "FarmFresh",
            "ingredients": ["pasteurized milk", "vitamin D3"]
        },
        "expected_missing": ["nutrition_info", "allergens", "certifications"]
    },
    {
        "sku_id": "SKU-003",
        "product_name": "Vitamin D3 Supplement",
        "brand": "HealthPlus",
        "existing_data": {
            "brand": "HealthPlus",
            "weight": "60 capsules"
        },
        "expected_missing": ["ingredients", "nutrition_info", "certifications", "fda_compliance"]
    }
]
```

### 8.2 Mock API Responses

```python
# Mock GDSN responses
MOCK_GDSN_DATA = {
    "SKU-001": {
        "ingredients": ["organic peanuts", "sea salt"],
        "nutrition_info": {
            "serving_size": "32g",
            "calories": 190,
            "total_fat": "16g",
            "protein": "7g",
            "carbohydrates": "7g"
        }
    },
    "SKU-002": {
        "nutrition_info": {
            "serving_size": "240ml",
            "calories": 150,
            "total_fat": "8g",
            "protein": "8g",
            "calcium": "30% DV"
        },
        "allergens": ["milk"]
    }
}

# Mock FDA compliance data
MOCK_FDA_DATA = {
    "SKU-003": {
        "fda_registered": True,
        "compliant": True,
        "warnings": [],
        "approved_claims": ["Supports bone health"]
    }
}
```

### 8.3 Sample PDF/Image Files

For the POC, include 2-3 sample files in `/data/sample_documents/`:
- `supplier_spec_sku001.pdf` - Product specification sheet
- `nutrition_label_sku002.png` - Nutrition facts image
- `certificate_organic.pdf` - Organic certification

These will be used by the Extraction Agent to demonstrate LLM-powered parsing.

---

## 9. Project Structure

```
product-enrichment-poc/
│
├── README.md
├── requirements.txt
├── .env.example
│
├── app/
│   ├── __init__.py
│   ├── main.py                    # Streamlit entry point
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                # Base agent class
│   │   ├── trigger.py             # Trigger Agent
│   │   ├── acquisition.py         # Acquisition Agent
│   │   ├── extraction.py          # Extraction Agent (LLM)
│   │   ├── validation.py          # Validation Agent
│   │   ├── mapping.py             # Mapping Agent
│   │   └── insight.py             # Insight Agent (LLM)
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py               # EnrichmentState definition
│   │   └── workflow.py            # LangGraph workflow definition
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── mock_products.py       # Sample product data
│   │   ├── mock_apis.py           # Mock GDSN, FDA, Supplier responses
│   │   └── sample_documents/      # Sample PDFs, images
│   │       ├── supplier_spec_sku001.pdf
│   │       └── nutrition_label_sku002.png
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   ├── product_db.py          # SQLite product database
│   │   └── snowflake_mock.py      # Mock Snowflake operations
│   │
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── components.py          # Reusable UI components
│   │   ├── pipeline_visual.py     # Pipeline visualization
│   │   └── styles.css             # Custom styling
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py              # Activity logging
│       └── quality_score.py       # Quality score calculation
│
├── tests/
│   ├── test_agents.py
│   ├── test_workflow.py
│   └── test_mock_data.py
│
└── docs/
    └── ARCHITECTURE.md            # This document
```

---

## 10. Dependencies

```text
# requirements.txt

# Core
python-dotenv==1.0.0
pydantic==2.5.0

# LangGraph & LLM
langgraph==0.0.40
langchain==0.1.0
langchain-openai==0.0.5
# OR langchain-anthropic==0.1.0

# UI
streamlit==1.29.0

# Data & Storage
sqlite3  # Built-in

# Document Processing (for Extraction Agent)
pypdf2==3.0.0
pillow==10.1.0

# Utilities
loguru==0.7.2

# Development
pytest==7.4.0
black==23.12.0
```

---

## 11. Environment Variables

```bash
# .env.example

# LLM Provider (choose one)
OPENAI_API_KEY=sk-...
# OR
ANTHROPIC_API_KEY=sk-ant-...

# Model Selection
LLM_MODEL=gpt-4o  # or claude-3-5-sonnet-20241022

# Optional: LangSmith Tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls-...
LANGCHAIN_PROJECT=product-enrichment-poc

# App Config
LOG_LEVEL=INFO
MOCK_DELAYS=true  # Simulate realistic API latencies
```

---

## 12. Execution Flow Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          END-TO-END EXECUTION                               │
└─────────────────────────────────────────────────────────────────────────────┘

1. USER ACTION
   └─► Select product from dropdown
   └─► Click "Start Enrichment"

2. TRIGGER AGENT (Pure Python, ~100ms)
   └─► Query mock product DB
   └─► Identify: ["ingredients", "allergens", "nutrition_info"]
   └─► Update state, emit log

3. ACQUISITION AGENT (Pure Python, ~2s simulated)
   └─► Call mock GDSN API → Get partial JSON
   └─► Call mock Supplier API → Get PDF path
   └─► Update state with acquired data

4. EXTRACTION AGENT (LLM, ~3s)
   └─► JSON data: Pass through (no LLM)
   └─► PDF data: Send to GPT-4o/Claude for extraction
   └─► Merge and structure all extracted attributes

5. VALIDATION AGENT (Hybrid, ~1.5s)
   └─► Rule checks: Completeness, format, FDA list
   └─► Calculate quality score
   └─► Flag issues/warnings

6. MAPPING AGENT (Pure Python, ~1s)
   └─► Merge extracted data with original (only missing fields)
   └─► Transform to Snowflake schema
   └─► Write to mock Snowflake (SQLite)

7. INSIGHT AGENT (LLM, ~2s)
   └─► Generate enrichment summary
   └─► Generate compliance report
   └─► Generate risk flags and recommendations

8. UI UPDATE
   └─► Display final enriched data
   └─► Show quality score
   └─► Show insights panel

TOTAL ESTIMATED TIME: ~10-12 seconds per SKU
```

---

## 13. Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Orchestration** | LangGraph | Industry standard, good observability, handles state well |
| **LLM usage** | Selective (only Extraction + Insight) | Cost/latency optimization, deterministic where possible |
| **UI** | Streamlit | Fast POC development, real-time updates possible |
| **Mock strategy** | In-memory + SQLite | No external dependencies for demo |
| **State management** | TypedDict flowing through graph | Type safety, easy debugging |
| **Logging** | Structured with timestamps | Required for activity log feature |

---

## 14. Future Enhancements (Post-POC)

For production, consider:
- **Real API integrations** (GDSN, FDA OpenFDA API)
- **Batch processing** (enrich 100s of SKUs)
- **Human-in-the-loop** for low-confidence extractions
- **Persistent storage** (PostgreSQL, actual Snowflake)
- **Authentication** for UI
- **Async execution** with job queuing
- **Retry logic** with exponential backoff
- **Caching** for repeated API calls

---

*Document Version: 1.0*
*Last Updated: January 2025*
