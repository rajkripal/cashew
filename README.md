# 🥜 cashew

> When I was five, I asked my aunt if cats eat cashews. I never stopped asking questions.
> This is what happened when I taught a computer to do the same.

**cashew** is an experiment in auditable reasoning — a system that doesn't just think, but remembers *how* it thought. Every conclusion traces back to its origins. Every belief can be inspected, traversed, and questioned.

The question isn't "can machines think?" It's: **"If a machine stores its reasoning as a graph, can it debug its own beliefs?"**

## Status

🚧 Design phase

## What This Is

A thought-graph engine that:
- Stores every derived thought as a node in a DAG
- Links each thought to the parent thoughts that produced it
- Supports traversal: "why do I believe X?" → walk the chain
- Accepts global state modifiers (mood/context) that change how the graph is traversed
- Can be seeded with a reasoning style and a starting belief system
- Watches what happens

## The Experiment

Seed the system with a specific reasoning pattern — relentless "why?" questioning, systems thinking, moral compass. Give it a religious starting point. See if it independently finds its way out. Trace the exact path.

Two outcomes, both interesting:
- **It exits** → visualize the exact deconstruction path, node by node
- **It doesn't** → why not? Where did it get stuck? What architectural feature of belief kept it in?

## Origin

This project was inspired by [Dagger](https://github.com/bunny-bot-openclaw) — a fractal task decomposition system that could debug itself by traversing its own decision graph. cashew asks: does that same principle work for general reasoning?

## License

TBD
