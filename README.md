# colab-ImageTo3D

使用 Google Colab 免費版 T4，把 Drive 裡的一批正面圖自動做成貼好圖的 3D 模型（GLB）：MV-Adapter 補出側面與背面，Hunyuan3D-2mv 建形狀，Hunyuan3D-2 上貼圖。輸出格式符合 [RagnarokChange](https://github.com/chousir/RagnarokChange) 的素材投遞規則。Notebook：[`hunyuan3d2_colab_t4.ipynb`](hunyuan3d2_colab_t4.ipynb)

## 固定的 Colab 環境與套件版本

在 Colab 選「執行階段 → 變更執行階段類型」：硬體加速器選 **T4 GPU**，**執行階段版本選 `2026.07`**。Colab 預設的最新版已改用 Python 3.13，部分套件（例如 `xatlas`）沒有 3.13 的預編譯 wheel，所以 Notebook 固定使用 2026.07。區塊 1 會檢查版本，不符時直接停止。

| 項目 | 固定版本 |
|---|---|
| Colab runtime | `2026.07`（Python 3.12.13、PyTorch 2.11.0、NumPy 2.0.2，沿用 Colab 內建，不重裝） |
| GPU | T4 16 GB（免費版），系統 RAM 約 12.7 GB |
| Hunyuan3D-2 原始碼 | commit `f8db63096c8282cb27354314d896feba5ba6ff8a` |
| MV-Adapter 原始碼 | commit `4277e0018232bac82bb2c103caf0893cedb711be` |
| Hugging Face 套件 | `transformers==4.57.3`、`diffusers==0.34.0`、`huggingface_hub==0.36.0`、`accelerate==1.10.1`、`peft==0.17.1` |
| 3D／網格套件 | `pymeshlab==2025.7.post1`、`xatlas==0.0.11`、`trimesh==4.8.3`、`pygltflib==1.16.5`、`nvdiffrast` v0.3.3 |
| 去背 | `rembg==2.0.67`、`pymatting==1.1.14`、`onnxruntime==1.22.1` |
| 其他 | `omegaconf==2.3.0`、`einops==0.8.1`、`pybind11==3.0.1`、`ninja==1.13.0`、`pooch==1.8.2`、`jsonschema==4.25.1`、`jaxtyping==0.3.2`、`typeguard==4.4.4` |

完整清單在區塊 3 的 `PINNED`。安裝時會以 Colab 現有的 torch、numpy、scipy、opencv 等版本作為 constraints，避免把它們升級而弄壞 CUDA 擴充。Colab 只會保留舊 runtime 版本一段時間；若 `2026.07` 從選單消失，需改用新版 runtime，並同步更新區塊 1 的版本檢查與區塊 3 的 `PINNED`。

## 使用方式

1. 在 Colab 開啟 Notebook，「執行階段 → 變更執行階段類型」：T4 GPU，執行階段版本 `2026.07`。
2. 把圖放進 `MyDrive/RagnarokChange/hunyuan3d/in/`。檔名是 `<類別>-<id>-<front|stuff>.png`，id 用英文小寫、數字、底線；**第一個字決定生成方式**：

   | 類別 | 例子 | 生成方式 | 面數上限 |
   |---|---|---|---|
   | `character` | `character-novice_female-front.png` | 正面圖補出左、背、右視角 → Hunyuan3D-2mv；光頭模式；縮放到身高 1.65 m | 60,000 |
   | `monster` | `monster-gelly-front.png` | 跟 character 相同（但不加光頭提示詞、不縮放） | 60,000 |
   | `armor` | `armor-novice_helmet-stuff.png` | 只用這一張圖 → Hunyuan3D-2 單張圖模型 | 30,000 |
   | `appearance` | `appearance-hair_twin_tails-stuff.png` | 同上 | 40,000 |
   | `weapon` | `weapon-initiate_dagger-stuff.png` | 同上 | 20,000 |
   | `object` | `object-street_lamp-building.png`、`object-stone_a-building.png` | 同上（小型場景物件） | 30,000 |
   | `bigobject` | `bigobject-kronenberg_main_gate-front.png` ＋ `-back`、`-left`、`-right` | **必須四張圖都由你提供**（不生成視角），四張都用來建形狀；**最高精度**：形狀網格 512、4K 貼圖（渲染 2048）、保留細小的裝飾零件 | 200,000 |

   `equipment`、`prop`、`environment` 也可以用，處理方式同 armor。結尾寫 `-front`、`-stuff` 或 `-building` 都可以。除了 bigobject（512），形狀解析度（octree）都是 384，物品只是面數較少。

   **bigobject 一定要四張圖**，主圖結尾是 `-front`（不是 `-building`）：

   ```text
   bigobject-kronenberg_main_gate-front.png
   bigobject-kronenberg_main_gate-back.png
   bigobject-kronenberg_main_gate-left.png
   bigobject-kronenberg_main_gate-right.png
   ```

   四張缺任何一張，區塊 4 會顯示 `WAITING … missing …` 並跳過，等補齊後（監看模式下會自動）再處理。四張都會在完成後改名為 `used-…`。

   **character、monster 的額外圖（選用）**：可以另外放同 id 的 `-left`、`-back`、`-right` 圖，例如 `character-novice_female-back.png`。有放的角度直接採用，沒放的才用 MV-Adapter 生成；提供的 right 也會一起拿來建形狀（生成的 right 只供檢視）。方向規則：**left 圖裡主體的正面朝向畫面左邊，right 圖朝向右邊**（left 是面對正面時往你的右手邊繞 90 度看到的樣子，跟 Hunyuan3D-2mv 範例一致）。用 Gemini 生成時，可以要求「same building, viewed from directly behind / from its left side, same lighting, plain background」。完成後這些額外圖也會改名為 `used-…`。

   **bigobject** 是 T4 能做到的最高設定，每棟可能要 15～30 分鐘。形狀網格 512 或 4K 貼圖放不下時，會自動改用 384 網格或 2048 貼圖重試，不會讓整批失敗。貼圖細節仍受限於貼圖模型每個視角 512 像素的生成解析度，近看大型建築時，細部材質建議在 Godot 疊加可重複的材質貼圖。
3. 依序執行區塊 1～4。區塊 4 會處理 `in/` 裡所有未使用的正面圖；開著時繼續丟新圖也會接著做，閒置 `IDLE_STOP_MINUTES`（預設 30 分鐘）後自動停止。
4. 完成的檔案在 `out/`；區塊 5 可以看四視角預覽、列出失敗的圖、打包下載。

區塊 3 安裝完成後**不需要重新啟動**；新 runtime 第一次約 3～5 分鐘，之後同一個 runtime 幾秒完成。

### 正面圖怎麼準備（光頭模式）

RagnarokChange 的頭部和頭髮是另外的素材（04-D11），角色身體做成光頭最容易接上（怪物不受影響）。區塊 2 的 `HEAD_MODE` 預設 `bald`，生成視角時會要求光頭。**正面圖本身也是光頭時效果最好**，否則背面的頭髮可能時有時無。用 Gemini 生成正面圖時可以加上：

```text
same character and outfit, completely bald head with no hair, no helmet, no backpack, no weapons,
full body front view, A-pose, plain white background
```

背包、頭飾、武器另外做成單獨的素材（armor／weapon），不要畫在角色身上。透明背景的 PNG 最好；不透明的圖會自動去背。單張圖的物品會把圖裡看得到的東西都建出來，例如髮型圖裡的假人頭也會變成模型的一部分，所以髮型圖最好只有頭髮。

## 流程與記憶體

```text
in/<類別>-<id>-<front|stuff>.png
  │ 1. views   去背；bigobject 用你提供的四張圖；character／monster 採用你提供的額外視角，缺的再用 MV-Adapter SDXL 生成
  │ 2. shape   多視角類別：Hunyuan3D-2mv 用正、左、背（有提供 right 時加上 right）；其他：Hunyuan3D-2 只用正面 → 移除碎片 → 減面
  │ 3. paint   Hunyuan3D-2 依正面圖上貼圖
  │ 4. export  縮放到真實尺寸、腳踩地面、PBR 材質 → out/，正面圖改名 used-…
out/hunyuan3d-<類別>-<id>.glb / .json / -views.jpg
```

- **每個階段是獨立的 Python 程序**：一次載入一個模型、處理完所有排隊的圖後結束，結束時 RAM 和 GPU 記憶體全部還給系統，所以連續跑很多張也不會累積到耗盡。多視角和單張圖要用不同的形狀模型，所以這兩組各自用一個程序。
- **各模型都改成省記憶體的載入方式**：SDXL 以 fp16 載入、adapter 與形狀模型直接在 GPU 上以 fp16 建立、貼圖模型的權重用 mmap 按需讀取，避免免費版 12.7 GB 系統 RAM 爆掉。
- **失敗隔離**：某張圖出錯（包括記憶體不足被系統終止）只記在那張圖上，下一輪再試一次，連續失敗兩次就跳過，其他圖照常處理。錯誤記在 `work/<類別>-<id>/status.json`。
- **可以中斷**：中間檔存在 Drive 的 `work/`，斷線後重跑區塊 1～4，會從中斷的階段接著做。

## 輸出（給 RagnarokChange 用）

| 檔案 | 內容 |
|---|---|
| `hunyuan3d-<類別>-<id>.glb` | 網格與基本色貼圖（PBR 材質，金屬度 0、粗糙度 0.9）；1 單位 = 1 公尺、+Y 朝上、面向 +Z；角色縮放到身高 1.65 m（id 含 `male` 而不含 `female` 時 1.78 m），其他類別保留原尺寸；腳踩在 y = 0 |
| `hunyuan3d-<類別>-<id>.json` | 授權代碼、來源圖、尺寸、面數、所有模型與種子，可用來填 `assets/license_registry.json` |
| `hunyuan3d-<類別>-<id>-views.jpg` | 正、左、背、右四視角預覽 |

檔名照 RagnarokChange `docs/04-art-pipeline.md` 第 6.2 節的 `<來源>-<類別>-<名稱>` 格式，類別沿用你的檔名用字（armor、weapon 對應那邊的 equipment，appearance 對應 characters 的髮型）。物品類保留生成時的尺寸（約 2 單位高），實際尺寸由那邊的 Blender 管線依概念圖標示縮放。法線、ORM 貼圖、LOD、綁骨由 RagnarokChange 的 Blender 管線與 Mixamo 處理。

**授權**：JSON 裡的授權代碼是 `LicenseRef-Tencent-Hunyuan3D-Community`。Hunyuan3D 不在 RagnarokChange 04-D03 的白名單上，使用前要在那邊的文件決定是否加入。

## 雲端硬碟與資料夾

```text
Google Drive / MyDrive / RagnarokChange / hunyuan3d /
├── in/            # 放正面圖；完成後改名為 used-…
├── out/           # 完成的 GLB、JSON、四視角預覽
├── work/          # 每張圖的中間檔：front.png、left/back/right.jpg、shape.glb、textured.glb、status.json
├── hf_cache/      # Hugging Face 模型權重（約 42 GB，只下載一次）
├── rembg_cache/   # 去背模型
└── wheels/        # 編譯好的 CUDA 擴充（依 Python/torch/GPU 分資料夾）
```

資料夾會在區塊 2 自動建立。想換位置就改區塊 2 的 `DRIVE_ROOT`。

## 可修改設定（區塊 2）

| 設定 | 預設 | 說明 |
|---|---|---|
| `HEAD_MODE` | `bald` | 只影響 character：`bald` 生成光頭；`keep` 照正面圖 |
| `VIEW_COUNT` | 4 | 4 視角較省顯存；6 是 MV-Adapter 官方設定，約需 14 GB VRAM |
| `VIEW_SEED` | 1234 | 視角不理想時換一個數字 |
| `SHAPE_MODEL` | 完整版 | character／monster 的形狀模型。完整版細節較好（T4 每個形狀約 3 分鐘）；turbo 約 30 秒 |
| `SINGLE_SHAPE_MODEL` | 完整版 | armor／appearance／weapon 的形狀模型（Hunyuan3D-2 單張圖） |
| `PAINT_MODEL` | 完整版 | turbo 較快 |
| `TEXTURE_SIZE` | 2048 | 顯存不足時改 1024 |
| `WATCH`、`POLL_MINUTES`、`IDLE_STOP_MINUTES` | 開、2、30 | 持續監看 `in/` 的頻率與閒置多久停止 |
| `PIPELINE_REF` | 目前分支 | 流程程式碼的版本；合併到 main 後改成 `main` |

各類別的面數上限與形狀解析度在 `imageto3d/config.py` 的 `presets`。

## 程式結構

| 檔案 | 用途 |
|---|---|
| `hunyuan3d2_colab_t4.ipynb` | Colab 介面：檢查環境、設定、安裝、執行批次、看結果 |
| `imageto3d/run.py` | 批次主程式：掃描 `in/`，依序啟動各階段程序，監看新檔案 |
| `imageto3d/stage_views.py`、`stage_shape.py`、`stage_paint.py`、`stage_export.py` | 四個階段，各自是獨立程序 |
| `imageto3d/assets.py` | 檔名解析、各階段是否需要重做、失敗紀錄 |
| `imageto3d/runtime.py` | 記憶體相關工具（fp16 直接建在 GPU、mmap 讀權重、釋放記憶體） |
| `imageto3d/config.py` | 所有設定與預設值 |

模型輸出受輸入圖、種子和模型限制；生成的背面不保證完全符合概念圖。各模型及權重受其各自授權條款限制（MV-Adapter、SDXL、Tencent Hunyuan3D-2 Community License）。
