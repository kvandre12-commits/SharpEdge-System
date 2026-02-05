ALTER TABLE open_resolution_regime ADD COLUMN setup_dir TEXT;     -- 'DOWN' | 'UP' | 'NONE'
ALTER TABLE open_resolution_regime ADD COLUMN key_source TEXT;    -- 'PRIOR_KEY_LOW' | 'PRIOR_KEY_HIGH' | 'FALLBACK_PM_OPEN'
