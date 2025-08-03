# Federal Register Document Selection and Ranking Criteria

## Environmental Regulations Priority Framework

### High Priority Environmental Topics
- **Air Quality Standards**: EPA regulations on emissions, particulate matter, ozone standards
- **Water Quality Protection**: Clean Water Act implementations, discharge permits, watershed protection
- **Climate Change Mitigation**: Carbon emissions regulations, renewable energy standards, climate adaptation
- **Toxic Substances Control**: Chemical safety regulations, pesticide controls, hazardous waste management
- **Environmental Justice**: Regulations affecting disadvantaged communities, cumulative impact assessments

### Geographic Priority Areas

#### New York State Focus Areas
- **New York City Metro**: Environmental regulations affecting NYC, Long Island, Westchester
- **Hudson River Valley**: Water quality, industrial cleanup, ecological restoration
- **Adirondack Region**: Wilderness protection, acid rain mitigation, forest management
- **Great Lakes Region**: Lake Ontario, Niagara River, binational environmental agreements
- **Environmental Justice Communities**: South Bronx, Brooklyn, Buffalo, Rochester disadvantaged areas

#### New Jersey State Focus Areas
- **Newark/Jersey City Metro**: Air quality, industrial pollution, port-related environmental issues
- **Delaware River Basin**: Water quality, chemical industry regulations, cross-border coordination
- **Pine Barrens**: Groundwater protection, ecosystem preservation, development restrictions
- **Shore Communities**: Coastal protection, sea level rise adaptation, storm water management
- **Environmental Justice Areas**: Camden, Trenton, Newark communities with cumulative pollution burdens

### Regulatory Impact Scoring

#### Immediate Impact (Score: 8-10)
- New regulations with compliance deadlines within 12 months
- Emergency environmental rules or temporary standards
- Regulations affecting major infrastructure projects (>$100M)
- Rules impacting public health with immediate implementation

#### High Impact (Score: 6-7)
- Regulations affecting multiple industry sectors
- Rules with significant economic impact (>$10M annually)
- Environmental standards affecting large populations (>100K people)
- Cross-state or regional environmental coordination requirements

#### Moderate Impact (Score: 4-5)
- Industry-specific environmental regulations
- Technical standard updates or clarifications
- Monitoring and reporting requirement changes
- Grant program announcements for environmental projects

#### Lower Priority (Score: 1-3)
- Procedural or administrative changes
- Minor technical corrections
- Routine permit renewals or extensions
- General informational notices

### Search Query Templates

#### Environmental Regulations by State
```
Query: "environmental regulation [STATE_NAME]"
Filters: {
  "agency": ["EPA", "Interior", "Commerce", "Transportation"],
  "chunk_type": ["rule", "preamble", "regulatory_text"]
}
```

#### Climate and Air Quality
```
Query: "climate change emissions air quality standards [REGION]"
Filters: {
  "agency": ["EPA"],
  "chunk_type": ["rule", "regulatory_text"]
}
```

#### Water Protection
```
Query: "water quality protection discharge permits [WATERSHED/RIVER]"
Filters: {
  "agency": ["EPA", "Interior"],
  "chunk_type": ["rule", "preamble"]
}
```

#### Environmental Justice
```
Query: "environmental justice disadvantaged communities cumulative impact"
Filters: {
  "chunk_type": ["preamble", "rule"],
  "agency": ["EPA", "HHS", "Transportation"]
}
```

### Content Analysis Keywords

#### High-Priority Environmental Keywords
- **Regulatory Actions**: "final rule", "proposed rule", "emergency regulation", "interim final rule"
- **Geographic Indicators**: "New York", "New Jersey", "tri-state", "metropolitan area", "regional"
- **Environmental Media**: "air quality", "water quality", "soil contamination", "groundwater"
- **Health Impacts**: "public health", "exposure", "risk assessment", "health-based standards"
- **Economic Impacts**: "cost-benefit", "economic analysis", "compliance costs", "small business"

#### Agency Priority Mapping
- **EPA**: Primary environmental regulator, highest relevance for environmental rules
- **Interior**: National parks, wildlife, federal lands affecting NY/NJ region
- **Commerce/NOAA**: Marine environmental protection, coastal zone management
- **Transportation**: Transportation-related environmental impacts, emissions standards
- **Energy**: Energy facility environmental reviews, renewable energy projects

### Ranking Algorithm Weights

#### Content Relevance (40%)
- Geographic mentions of NY/NJ: +20 points
- Environmental keywords density: +15 points
- Regulatory action type: +5 points

#### Agency Authority (25%)
- EPA regulations: +25 points
- Interior/Commerce environmental rules: +15 points
- Other agencies with environmental components: +10 points

#### Impact Scope (20%)
- Multi-state/regional impact: +20 points
- Single state impact: +15 points
- Local/municipal impact: +10 points

#### Urgency/Timeline (15%)
- Immediate implementation: +15 points
- Implementation within 1 year: +10 points
- Future implementation (>1 year): +5 points

### Example High-Priority Scenarios

1. **EPA Air Quality Standard for NYC Metro**
   - Score: 9-10
   - Rationale: Direct health impact, large population, immediate compliance required

2. **Delaware River Basin Water Quality Rules**
   - Score: 8-9
   - Rationale: Cross-state coordination, critical water supply, environmental justice implications

3. **Climate Resilience Requirements for Coastal NJ**
   - Score: 7-8
   - Rationale: Sea level rise adaptation, infrastructure protection, long-term planning

4. **Superfund Site Cleanup Standards (Hudson River)**
   - Score: 8-9
   - Rationale: Major contamination site, multi-stakeholder impact, health protection

### Search Strategy Implementation

#### Daily Monitoring Queries
1. Environmental regulations + "New York" OR "New Jersey"
2. EPA rules + air quality + tri-state
3. Water quality + Delaware River + Hudson River
4. Environmental justice + disadvantaged communities + NY + NJ
5. Climate change + coastal + adaptation + New Jersey

#### Weekly Deep Dive Queries
1. Comprehensive agency-specific searches (EPA, Interior, Commerce)
2. Cross-reference with state environmental agency actions
3. Industry-specific environmental compliance updates
4. Federal funding opportunities for environmental projects

This framework ensures systematic identification and prioritization of Federal Register documents that have the highest relevance and impact for environmental protection in the New York and New Jersey region.
