# Resource Permission Mining Implementation Notes

**Overview**
Resource permission mining based on **OrdinoR (2022)** framework.

## 1. Configuration

- **Execution Context**: CT + AT + TT
    - **CT**: Case attribute `LoanGoal` (loan purpose, 14 categories)
    - **AT**: Activity name
    - **TT**: Day of week
- **Resource Clustering**: AHC, Ward's linkage, 10 groups
- **Group Profiling**: Full Recall (default) or OverallScore ($\omega_1=0.5, p=0.5$)

### Why LoanGoal (Not Trace Clustering)?

> From OrdinoR paper Section 6.2: *"For CT+AT+TT (case attribute), we selected [...] the attribute recording the loan purpose of applicants for log BPIC17."*
>
> **Rationale**: Trace clustering requires a complete trace, unavailable at simulation start. `LoanGoal` is known at case instantiation.

---

## 2. Pipeline

1. **Data Prep**: Filter completed events, exclude system users (caveat: 2 activities don't have a completed event, W_Personal Loan collection and W_Shortened completion)
2. **Case Types**: Extract from `case:LoanGoal` column
3. **Execution Contexts**: Combine CT+AT+TT
4. **Resource Profiling**: `ordinor.resource_features.direct_count()`
5. **Resource Clustering**: `ordinor.group_discovery.ahc()`
6. **Group Profiling**: `ordinor.group_profiling.overall_score()`

---

## 3. Usage

```python
allocator.allocate(
    activity="A_Create Application",
    timestamp=dt,
    case_type="Home improvement"  # One of 14 LoanGoal values
)
```

**Valid `case_type` values** (LoanGoal):
- Existing loan takeover, Home improvement, Car, Other, Remaining debt home, Not specified, Unknown, Caravan/Camper, Tax payments, Extra spending limit, Motorcycle, Boat, Business goal, Debt restructuring

```python
from resources.resource_allocation import ResourceAllocator

# Initialize once at the start of simulation
resource_allocator = ResourceAllocator(
    log_path="path/to/BPIC2017.xes", # Only needed if cache missing or for availability
    permission_method='ordinor',
    cache_path="resources/BPIC2017/resource_permissions/ordinor_fullrecall.pkl"
)

# In the simulation loop:
selected_resource = resource_allocator.allocate(
    activity="A_Create Application",
    timestamp=current_simulation_time,
    case_type="Home improvement" # Must be provided for best results
)
```

---

## 4. Performance

- **Discovery**: 15-60 min (full BPIC2017)
- **Allocation**: <1ms
- **Caching**: Model pickled to `ordinor_fullrecall.pkl`