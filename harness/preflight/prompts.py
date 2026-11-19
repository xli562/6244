COMMON = """Write one compact reusable Python run(api) that handles unseen household manipulation instructions.
The function reads api.instruction, api.catalog() (dict: name -> {'kind':item/fixture/region,'parent':name or None}),
api.pos(name) (live xyz in metres, numpy), api.remaining() (starts at 500). Map words to scene entities,
including spatial descriptions, and execute all requested clauses in order. Avoid task-ID or full instruction lookup.
Resolve ambiguous instances based on scene observations. Ground the strongest initial choices you can.
Only documented methods, numpy/math imports, no private attributes. Return one fenced python block.
Use the shared physical skill library via api.act(step) with this dictionary IR:
 place: {'op':'place','item':picked_name,'target':picked_name,'grasps':['standard']} (up to three ordered grasps
 from standard, rim [pinch a rim], thin_edge [align to a thin upright object]; retry a different grasp after no lift).
 drawer: {'op':'open' or 'close','target':named drawer region}; do not open a destination unless needed.
 contact stroke: {'op':'stroke','target':named entity,'offset':[dx,dy,dz],'delta':[dx,dy,dz]}.
 This closes the fingers, approaches from above, then pushes along delta; both vectors in world metres, norm <= .30.
The receipt returns executed, remaining, steps_used, and for an executed step observed watch position change
(delta_xyz), final xy_gap and max_lift when measurable. Re-check positions and budget before another action.
Do not treat an empty source or target as a default first object. Prefer a short program that can finish within
one CLI call. The whole program freezes before the evaluator selects the new tasks. No deployment model calls."""
DIRECT = """You have feedback after the action. Give your best grounded nominations directly in item and target.
Use the public catalogue and positions freely when deciding. Do not call preview or add an expect field."""
CHECKED = """In addition to your initial item/target nominations, declare independent semantic constraints
for EACH role in step['expect']. Each role selector is {'kind':'item'|'region'|'fixture','terms':['word',...],
'parent':optional_exact_observed_parent_name, 'near':optional_visible_anchor_name} OR use 'axis':'x'|'y'|'z',
'rank':'min'|'max'|'median' instead of near. All terms must be alphabetic tokens occurring literally in candidate
names (underscore splits words); no instance IDs or serial numbers. Map natural-language synonyms to the words in
the scene. Terms and owners should express the instruction, not just copy your pick and rationalize it. When an
observed location label describes a source, prefer its physical anchor to assuming a world-coordinate convention.
api.preview(step) reads live scene evidence, takes zero control steps; returns None when the checker is disabled,
otherwise role -> {pick,resolved,status,...}. status is confirmed/corrected/unresolved. You may refine your intent
when constraints fail. api.act rechecks both roles immediately before moving, automatically substitutes the unique
match, and returns executed=False for unresolved or tied evidence. Never erase a genuine instruction constraint
to make a tie pass. Always include a strong direct nomination even if checking is disabled. No special cases for
practice instructions. Fictional example: expect={'item':{'kind':'item','terms':['block'],'near':'desk_rear_mark'},
'target':{'kind':'region','terms':['inner'],'parent':'bin_1'}}. Use real observed names at runtime."""
