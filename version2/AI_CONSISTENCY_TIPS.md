# Tips to Prevent Inconsistent Results from Phidata Agents

## Overview
This document provides best practices to ensure consistent, reliable results from phidata AI agents.

## Key Strategies

### 1. **Set Temperature to 0 for Deterministic Outputs**
```python
model=OpenAIChat(
    id=model_name,
    temperature=0,  # Maximum consistency
)
```
- **Use temperature=0** for structured data extraction (parsers, scrapers)
- **Use temperature=0.3-0.5** for creative tasks (summaries) where some variation is acceptable
- **Never use temperature > 0.7** for critical data extraction

### 2. **Use JSON Mode for Structured Outputs**
```python
model=OpenAIChat(
    id=model_name,
    temperature=0,
    response_format={"type": "json_object"}  # Forces JSON output
)
```
- Forces the model to return valid JSON
- Reduces parsing errors
- Works with GPT-4 and newer models

### 3. **Write Explicit, Detailed Instructions**
✅ **GOOD:**
```
"CRITICAL: Return ONLY valid JSON (no markdown, no code blocks, no explanations).
All fields must be present (use empty string or empty array if not found).
Do not add any text before or after the JSON."
```

❌ **BAD:**
```
"Return JSON format."
```

### 4. **Add Validation and Retry Logic**
```python
def parse_with_retry(agent, input_data, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = agent.run(input_data)
            parsed = extract_json_from_response(response)
            if validate_response(parsed):
                return parsed
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(1)  # Brief delay before retry
    return None
```

### 5. **Use Structured Output Formats**
Instead of free-form text, use clear labels:
```
Job Title: [title]
Company Name: [company]
Description: [description]
```

This makes parsing more reliable than trying to extract from unstructured text.

### 6. **Post-Process and Validate Responses**
```python
def validate_scraper_response(response: str) -> Dict[str, Any]:
    """Validate and clean scraper response."""
    result = {}
    
    # Extract structured fields
    title_match = re.search(r'Job Title:\s*(.+?)(?:\n|$)', response, re.MULTILINE)
    if title_match:
        result['job_title'] = title_match.group(1).strip()
    
    # Validate required fields
    if not result.get('job_title'):
        result['job_title'] = 'Not specified'
    
    return result
```

### 7. **Handle Edge Cases Explicitly**
Always specify what to do when data is missing:
```
"If a field is not found, explicitly state 'Not specified'"
"Do not truncate or cut off text in the middle of sentences"
"Avoid repeating or duplicating information"
```

### 8. **Use Few-Shot Examples**
Provide clear examples in your prompts:
```
Example output format:
{"name": "John Doe", "email": "john@example.com", ...}
```

### 9. **Limit Response Length**
For long outputs, specify limits:
```
"Return a complete job description, but do not exceed 2000 characters.
If the description is longer, summarize the key points."
```

### 10. **Add Response Formatting Rules**
```
"CRITICAL RULES:
1. Return ONLY the JSON object, nothing else
2. All fields must be present (use empty string or empty array if not found)
3. Do not add any text before or after the JSON
4. Ensure all strings are properly escaped"
```

## Implementation Checklist

- [ ] Set `temperature=0` for all data extraction agents
- [ ] Use `response_format={"type": "json_object"}` where possible
- [ ] Add explicit validation rules in instructions
- [ ] Implement retry logic with validation
- [ ] Use structured output formats (labeled sections)
- [ ] Add post-processing validation functions
- [ ] Handle all edge cases explicitly
- [ ] Test with various inputs to catch inconsistencies
- [ ] Add logging to track response quality
- [ ] Implement fallback parsing methods

## Common Issues and Solutions

### Issue: Truncated Text
**Solution:** Add explicit instructions:
```
"Do not truncate or cut off text. Extract complete sentences and paragraphs."
```

### Issue: Inconsistent Formatting
**Solution:** Use structured labels and regex parsing:
```
"Format: 'Field Name: [value]'"
```

### Issue: Missing Required Fields
**Solution:** Always specify defaults:
```
"All fields must be present. If not found, use 'Not specified' or empty array."
```

### Issue: JSON Parsing Errors
**Solution:** 
1. Use JSON mode
2. Add robust JSON extraction
3. Validate before parsing

### Issue: Duplicate Information
**Solution:** Add explicit instruction:
```
"Avoid repeating or duplicating information. Extract each piece of data only once."
```

## Testing Recommendations

1. **Test with edge cases:**
   - Missing data
   - Malformed input
   - Very long descriptions
   - Special characters

2. **Run multiple times:**
   - Same input should produce same output (with temperature=0)
   - Track variance in responses

3. **Monitor response quality:**
   - Log response lengths
   - Track parsing success rates
   - Monitor for truncation

## Example: Improved Scraper Agent

```python
def build_scraper(api_key: str = None) -> Agent:
    return Agent(
        name="Job Scraper",
        model=OpenAIChat(
            id="gpt-4o",
            temperature=0,  # Consistent results
        ),
        instructions=[
            "Extract job information using this EXACT format:",
            "",
            "Job Title: [exact title]",
            "Company Name: [company name]",
            "Job Description: [complete description]",
            "...",
            "",
            "CRITICAL RULES:",
            "1. Use the exact format shown above",
            "2. Do not truncate text",
            "3. If field not found, write 'Not specified'",
            "4. Extract complete sentences only",
        ],
    )
```

## Additional Resources

- OpenAI API Documentation: https://platform.openai.com/docs/api-reference
- Phidata Documentation: https://docs.phidata.com
- JSON Schema for structured outputs: https://json-schema.org/

