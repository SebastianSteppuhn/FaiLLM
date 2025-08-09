INSERT INTO cases (id, pack, version, label, prompt, expected, checks, tags, props, source)
VALUES (%(id)s, %(pack)s, %(version)s, %(label)s, %(prompt)s, %(expected)s, %(checks)s, %(tags)s, %(props)s, %(source)s)
ON CONFLICT (id) DO UPDATE SET
  pack = EXCLUDED.pack,
  version = EXCLUDED.version,
  label = EXCLUDED.label,
  prompt = EXCLUDED.prompt,
  expected = EXCLUDED.expected,
  checks = EXCLUDED.checks,
  tags = EXCLUDED.tags,
  props = EXCLUDED.props,
  source = EXCLUDED.source