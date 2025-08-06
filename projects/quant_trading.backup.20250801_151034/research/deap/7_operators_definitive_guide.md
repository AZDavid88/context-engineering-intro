# DEAP Operators - Definitive Guide from Official Documentation

**Source**: 
- https://deap.readthedocs.io/en/master/api/tools.html
- https://deap.readthedocs.io/en/master/tutorials/basic/part2.html
**Extraction Date**: 2025-01-25
**Method**: Jina API scraping of official DEAP documentation
**Research Gap Addressed**: Missing operator documentation that caused Individual class preservation failures

## CRITICAL FINDING: Operator Behavior Patterns

### Key Operator Principles (From Official Documentation)

1. **Mutation "only mutates"** - requires independent copy if original must be kept
2. **Crossover "only mates"** - requires independent copies if originals must be kept  
3. **Selection returns references** - not duplicates of individuals
4. **Operators modify individuals in-place**
5. **Fitness must be manually invalidated after modifications**

### Built-in Crossover Operators (Verified from Documentation)

#### cxTwoPoint(ind1, ind2)
- **Parameters**: Two individuals
- **Returns**: A tuple of two modified individuals  
- **Behavior**: Exchanges segments between two individuals at two random points
- **Individual Class**: Preserved (operates in-place on Individual objects)

#### cxUniform(ind1, ind2, indpb)
- **Parameters**: Two individuals, independent probability
- **Returns**: A tuple of two modified individuals
- **Behavior**: Swaps attributes between individuals based on probability
- **Individual Class**: Preserved (operates in-place on Individual objects)

#### cxOnePoint(ind1, ind2)
- **Parameters**: Two individuals
- **Returns**: A tuple of two modified individuals
- **Behavior**: Randomly select crossover point and exchange each subtree
- **Individual Class**: Preserved (operates in-place on Individual objects)

### Built-in Mutation Operators (Verified from Documentation)

#### mutGaussian(individual, mu, sigma, indpb)
- **Parameters**: Individual, mean, standard deviation, mutation probability
- **Returns**: A tuple containing the mutated individual
- **Behavior**: Applies Gaussian mutation to individual attributes
- **Individual Class**: Preserved (operates in-place on Individual object)

#### mutShuffleIndexes(individual, indpb)
- **Parameters**: Individual, mutation probability  
- **Returns**: A tuple containing the mutated individual
- **Behavior**: Randomly shuffles individual's attributes
- **Individual Class**: Preserved (operates in-place on Individual object)

#### mutFlipBit(individual, indpb)
- **Parameters**: Individual, mutation probability
- **Returns**: A tuple containing the mutated individual
- **Behavior**: Flips boolean attributes with given probability
- **Individual Class**: Preserved (operates in-place on Individual object)

## Official Usage Pattern (From Documentation)

### Correct Operator Usage
```python
# OFFICIAL PATTERN: Clone before applying operators
mutant = toolbox.clone(ind1)
ind2, = tools.mutGaussian(mutant, mu=0.0, sigma=0.2, indpb=0.2)
del mutant.fitness.values  # CRITICAL: Invalidate fitness after mutation
```

### Fitness Handling (From Documentation)
```python
# Fitness is a tuple of values
def evaluate(individual):
    return sum(individual),  # Must return tuple

# Set fitness values
ind1.fitness.values = evaluate(ind1)
print(ind1.fitness.valid)  # True after setting values

# Invalidate fitness after genetic operations
del child1.fitness.values  # Makes fitness.valid = False
del child2.fitness.values
```

## Critical Implementation Insight

### Why Built-in Operators Work (Evidence-Based)
- **In-place modification**: Built-in operators modify the Individual object directly
- **Type preservation**: Individual class and all attributes (including fitness) are maintained
- **Return pattern**: Return the same Individual objects, not plain lists

### Why Custom Operators Failed (Evidence-Based)
- **Created new objects**: `new_ind1 = ind1.copy()` creates plain list, not Individual
- **Lost class type**: Plain list has no `fitness` attribute
- **Broke DEAP algorithms**: eaSimple expects Individual objects with fitness attributes

## Verified Working Pattern

### Toolbox Registration (Evidence-Based)
```python
from deap import base, creator, tools

# Create Individual class  
creator.create("Individual", list, fitness=creator.FitnessMax)

# Register BUILT-IN operators (WORKS)
toolbox = base.Toolbox()
toolbox.register("mate", tools.cxTwoPoint)  # ✅ Preserves Individual class
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)  # ✅ Preserves Individual class
```

### Verification Test (Evidence-Based)
```python
# Test Individual class preservation
ind1 = creator.Individual([0.1, 0.2, 0.3])
ind2 = creator.Individual([0.4, 0.5, 0.6])

# Test crossover
offspring = tools.cxTwoPoint(ind1, ind2)
print(type(offspring[0]))  # <class 'deap.creator.Individual'> ✅
print(hasattr(offspring[0], 'fitness'))  # True ✅

# Test mutation
mutant = tools.mutGaussian(ind1, mu=0, sigma=0.1, indpb=0.1)
print(type(mutant[0]))  # <class 'deap.creator.Individual'> ✅
print(hasattr(mutant[0], 'fitness'))  # True ✅
```

## Research Conclusion

**Evidence**: Built-in DEAP operators preserve Individual class because they:
1. Operate in-place on the Individual object
2. Return the same Individual objects (modified)
3. Maintain all Individual attributes including fitness

**Solution**: Always use DEAP's built-in operators unless specific algorithmic requirements cannot be met with existing operators.

**Verification**: This solution was tested and confirmed to fix the "list object has no attribute 'fitness'" error in our genetic engine.