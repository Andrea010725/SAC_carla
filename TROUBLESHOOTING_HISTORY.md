# CARLA Training Troubleshooting History

## Problem Summary

Training consistently got stuck at around 160 steps during `env.reset()` → `sync_mode.tick()` call.

## Root Cause Analysis

The original codebase was working correctly. Issues arose from:

1. **Config Conflict**: `config.py` had two conflicting render settings:
   - Line 13: `self.render = False`
   - Line 74: `self.render = True` (overwrites line 13)

2. **Unknown Environmental Change**: The 160-step hang suggests something in the environment changed (CARLA version, system config, network settings, etc.) rather than a code issue.

## Failed Fix Attempts

### Attempt 1: Sensor Cleanup & Timeout Adjustments
- Modified sensor cleanup order in `carla_env.py`
- Increased timeout parameters (client: 60→120s, tick: 10→15s)
- **Result**: Still stuck at 160 steps

### Attempt 2: Watchdog & Force Restart
- Added watchdog monitoring thread
- Added force restart every 1024 steps
- Added episode-based CARLA restart (every 50 episodes)
- **Result**: Pygame failed to start

### Attempt 3: Disable Sync Mode
- Temporarily disabled `CarlaSyncMode`
- **Result**: Training ran too fast (0.2s/episode), CARLA black screen, physics broken

## Final Solution: Git Restore

Rolled back to original working version:

```bash
git restore carla_base/carla_env.py carla_base/carla_sync_mode.py config.py
```

### Restored State:
- ✅ pygame enabled (`config.py` line 74: `self.render = True`)
- ✅ Original sync_mode logic intact
- ✅ Standard sensor cleanup

## B2Drive Scenarios Analysis

Analyzed 10 scenario files (3,125 lines total) from `b2drive_scenarios/`:
- All use ScenarioRunner framework
- Complex actor behaviors, traffic patterns, signal control
- **Recommendation**: Gradual lightweight migration approach (extract waypoints, obstacle patterns, traffic configs)

See `B2DRIVE_SCENARIOS_ANALYSIS.md` for detailed analysis.

## Expected Behavior After Restore

If original code works:
- ✅ pygame window starts
- ✅ CARLA front camera view visible
- ✅ HUD displays Planner info
- ✅ Training progresses normally

If 160-step hang reappears:
- ❌ Problem is environmental, not code-related
- Need to investigate: CARLA version, system updates, network changes, memory pressure

## Diagnostic Commands

```bash
# Clean environment
pkill -9 python
pkill -9 CarlaUE4
sleep 3

# Start CARLA manually
cd /home/ajifang/carla
./CarlaUE4.sh -RenderOffScreen -carla-server -benchmark -fps=20 -quality-level=Low -nosound

# In another terminal, run training
cd /home/ajifang/SAC_carla
./start_ppo_training_wandb.sh
```

## Key Files

- **`carla_base/carla_env.py`** (1,155 lines): Gym environment wrapper
  - Lines 130-135: pygame initialization
  - Lines 211-371: `reset()` method (where hang occurs)
  - Lines 422-440: `_tick_once()` method

- **`carla_base/carla_sync_mode.py`** (112 lines): Synchronization manager
  - Lines 47-51: `tick()` method (blocking point)

- **`config.py`** (114 lines): Configuration
  - Line 74: `self.render = True` (critical for pygame)

## Next Steps

1. Run training with restored code
2. If hang persists, check CARLA logs for server-side issues
3. Verify CARLA version matches expected 0.9.15
4. Check system resources during hang (CPU, memory, network)
