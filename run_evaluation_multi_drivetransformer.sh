BASE_PORT=30000
BASE_TM_PORT=50000
IS_BENCH2DRIVE=True
BASE_ROUTES=leaderboard/data/bench2drive220
TEAM_AGENT=team_code/drivetransformer_b2d_agent_diffusion_small_mlp_w_es.py

# TEAM_CONFIG=DriveTransformer/adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large_w_diffusion_head_small_mlp.py+/lustre/scratch/ezhu3/drive_transformer_work_dir/latest.pth
TEAM_CONFIG=DriveTransformer/adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large_w_diffusion_head_small_mlp.py+/zfsauton2/home/ezhu3/Bench2Drive/lastest_diffusion_pth

BASE_CHECKPOINT_ENDPOINT=eval_bench2drive220
PLANNER_TYPE=only_traj
ALGO=DriveTransformerAgentDiffusion_Small_MLP_W_ES
SAVE_PATH=/home/scratch/ezhu3/eval_bench2drive220_${ALGO}_${PLANNER_TYPE}

export PYTHONPATH="/zfsauton2/home/ezhu3/Bench2Drive/DriveTransformer:${PYTHONPATH}"

if [ ! -d "${ALGO}_b2d_${PLANNER_TYPE}" ]; then
    mkdir ${ALGO}_b2d_${PLANNER_TYPE}
    echo -e "\033[32m Directory ${ALGO}_b2d_${PLANNER_TYPE} created. \033[0m"
else
    echo -e "\033[32m Directory ${ALGO}_b2d_${PLANNER_TYPE} already exists. \033[0m"
fi

# Check if the split_xml script needs to be executed
if [ ! -f "${BASE_ROUTES}_${ALGO}_${PLANNER_TYPE}_split_done.flag" ]; then
    echo -e "****************************\033[33m Attention \033[0m ****************************"
    echo -e "\033[33m Running split_xml.py \033[0m"
    TASK_NUM=4
    python tools/split_xml.py $BASE_ROUTES $TASK_NUM $ALGO $PLANNER_TYPE
    touch "${BASE_ROUTES}_${ALGO}_${PLANNER_TYPE}_split_done.flag"
    echo -e "\033[32m Splitting complete. Flag file created. \033[0m"
else
    echo -e "\033[32m Splitting already done. \033[0m"
fi

# ============================================================================
# MODIFICATION: Set empty DISPLAY for headless operation (requires -opengl flag in CARLA)
# ============================================================================
export DISPLAY=
echo -e "\033[32m DISPLAY set to empty (headless mode) \033[0m"
echo -e "\033[33m Note: This requires CARLA to be started with -opengl flag \033[0m"
# ============================================================================
# END MODIFICATION
# ============================================================================

echo -e "**************\033[36m Please Manually adjust GPU or TASK_ID \033[0m **************"
# GPU_RANK_LIST=(0 1 2 3 4 5 6 7)
# TASK_LIST=(0 1 2 3 4 5 6 7)
GPU_RANK_LIST=(0 1 2 3)
TASK_LIST=(0 1 2 3)
echo -e "\033[32m GPU_RANK_LIST: $GPU_RANK_LIST \033[0m"
echo -e "\033[32m TASK_LIST: $TASK_LIST \033[0m"
echo -e "***********************************************************************************"

length=${#GPU_RANK_LIST[@]}
for ((i=0; i<$length; i++ )); do
    PORT=$((BASE_PORT + i * 150))
    TM_PORT=$((BASE_TM_PORT + i * 150))
    ROUTES="${BASE_ROUTES}_${TASK_LIST[$i]}_${ALGO}_${PLANNER_TYPE}.xml"
    CHECKPOINT_ENDPOINT="${ALGO}_b2d_${PLANNER_TYPE}/${BASE_CHECKPOINT_ENDPOINT}_${TASK_LIST[$i]}.json"
    GPU_RANK=${GPU_RANK_LIST[$i]}
    echo -e "\033[32m ALGO: $ALGO \033[0m"
    echo -e "\033[32m PLANNER_TYPE: $PLANNER_TYPE \033[0m"
    echo -e "\033[32m TASK_ID: $i \033[0m"
    echo -e "\033[32m PORT: $PORT \033[0m"
    echo -e "\033[32m TM_PORT: $TM_PORT \033[0m"
    echo -e "\033[32m CHECKPOINT_ENDPOINT: $CHECKPOINT_ENDPOINT \033[0m"
    echo -e "\033[32m GPU_RANK: $GPU_RANK \033[0m"
    echo -e "\033[32m bash leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK \033[0m"
    echo -e "***********************************************************************************"
    # ============================================================================
    # MODIFICATION: Use restart wrapper to handle CARLA crashes
    # ============================================================================
    # DISPLAY= bash -e leaderboard/scripts/run_evaluation_original.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK 2>&1 > ${BASE_ROUTES}_${TASK_LIST[$i]}_${ALGO}_${PLANNER_TYPE}.log &
    LOG_FILE_PATH="${BASE_ROUTES}_${TASK_LIST[$i]}_${ALGO}_${PLANNER_TYPE}.log"
    nohup bash leaderboard/scripts/run_evaluation_with_restart.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK "$LOG_FILE_PATH" > "$LOG_FILE_PATH" 2>&1 &
    # ============================================================================
    # END MODIFICATION
    # ============================================================================
    # source leaderboard/scripts/run_evaluation_original.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK

    sleep 35
done
wait