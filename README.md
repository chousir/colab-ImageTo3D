# colab-ImageTo3D

使用 Google Colab 免費版 T4，從一張正面概念圖生成多視角參考圖，再建立並貼圖 Hunyuan3D GLB。Notebook：[`hunyuan3d2_colab_t4.ipynb`](hunyuan3d2_colab_t4.ipynb)

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
| 其他 | `omegaconf==2.3.0`、`einops==0.8.1`、`pybind11==3.0.1`、`ninja==1.13.0`、`pooch==1.8.2`、`jsonschema==4.25.1` |

完整清單在區塊 3 的 `PINNED`。安裝時會以 Colab 現有的 torch、numpy、scipy、opencv 等版本作為 constraints，避免把它們升級而弄壞 CUDA 擴充。Colab 只會保留舊 runtime 版本一段時間；若 `2026.07` 從選單消失，需改用新版 runtime，並同步更新區塊 1 的版本檢查與區塊 3 的 `PINNED`。

## 雲端硬碟與資料夾

Notebook 預設使用：

```text
Google Drive / MyDrive / RagnarokChange / hunyuan3d /
├── in/                  # 放正面圖；生成的左右、背面圖也先存在這裡
├── out/                 # 完成的 GLB 與 JSON metadata
├── hf_cache/            # Hugging Face 模型權重快取
├── rembg_cache/         # 去背用 U2Net 權重快取
└── wheels/              # 編譯好的 custom_rasterizer wheel 與 mesh_processor（依 Python/torch/GPU 分資料夾）
```

**不需要預先建立這些資料夾。** 第一次執行 Notebook 區塊 2、掛載 Google Drive 後，程式會自動建立它們。之後把正面圖放進 `MyDrive/RagnarokChange/hunyuan3d/in` 即可。若想換位置，先修改區塊 2 的 `DRIVE_ROOT`。

模型權重會在流程首次用到時自動下載到 Drive，不必手動下載或解壓，也不必每次重新下載。區塊 2 會檢查快取及可用空間；估計約需 22 GB，實際用量視模型版本與已快取內容而異。你有 5 TB 空間，足以容納這些模型與輸出。請以 Google Drive 顯示的實際剩餘空間為準。

Colab 的執行環境則不同：Hunyuan3D/MV-Adapter 程式碼與 Python 套件安裝在 Colab runtime，runtime 回收後需由區塊 3 重新安裝（固定版本，約 3～5 分鐘）。最花時間的 CUDA 擴充（`custom_rasterizer`、`mesh_processor`）只在第一次編譯，之後從 Drive 的 `wheels/` 直接安裝；模型權重也直接從 Drive 快取讀取。新 runtime 安裝完成後區塊 3 會自動重啟 runtime。

## 會下載哪些模型

只會下載目前設定和執行流程實際使用的模型，不會把下表所有選項全部下載。不同名稱是因為各模型負責不同工作，另外也有品質、速度和顯存取捨。

| 用途 | 預設／可選模型 | 何時載入 |
|---|---|---|
| 正面圖生成左右、背面 | [MV-Adapter](https://huggingface.co/huanngzh/mv-adapter) image-to-multiview SDXL adapter | 第一次執行區塊 5 |
| 多視角影像基礎模型 | [SDXL Base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)（只下載 fp16 權重） | 第一次執行區塊 5 |
| SDXL 專用 VAE | [`madebyollin/sdxl-vae-fp16-fix`](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) | 第一次執行區塊 5 |
| 多視角形狀 | [Hunyuan3D-2mv](https://huggingface.co/tencent/Hunyuan3D-2mv) 的 shape checkpoint；預設 `hunyuan3d-dit-v2-mv-turbo`，也可選完整模型 | 第一次執行區塊 6 |
| 形狀加速解碼 | [Hunyuan3D-2](https://huggingface.co/tencent/Hunyuan3D-2) 的 `hunyuan3d-vae-v2-0-turbo`（FlashVDM） | 使用 turbo 且 `SHAPE_FLASHVDM = True` 時，第一次執行區塊 6 |
| 產生貼圖 | [Hunyuan3D-2](https://huggingface.co/tencent/Hunyuan3D-2) 的 paint checkpoint；預設完整模型，也可選 turbo | 第一次執行區塊 7 |
| 自動去背 | rembg/U2Net 權重 | 首次處理沒有透明背景的輸入圖時；快取在 `rembg_cache/` |

選項含義：

- `VIEW_COUNT = 4`（預設）：只生成 front/left/back/right，較省 T4 顯存。改為 `6` 是 MV-Adapter 官方的 6 視角設定，約需 14 GB GPU VRAM；若 OOM 改回 `4`，再重跑區塊 2、4、5。原本的 SD 2.1 備案已移除：`stabilityai/stable-diffusion-2-1-base` 已從 Hugging Face 下架。
- `SHAPE_MODEL` 的完整模型較慢、步數較多；`...-turbo` 較快。兩者都是 Hunyuan 多視角形狀模型。
- `SHAPE_FLASHVDM`：turbo 形狀模型使用官方 FlashVDM 加速解碼；若出錯改為 `False`。
- `PAINT_MODEL` 的完整模型較慢；`...-turbo` 較快。

程式會在下載前嘗試檢查 Drive 可用空間；若檢查結果與 Drive 網頁顯示不一致，以 Drive 帳戶的實際配額為準。Hugging Face 若要求存取授權，請先在模型頁接受條款，並可在 Colab Secrets 設定 `HF_TOKEN`。

## 輸入檔名

區塊 4 會從 `in/` 自動找尚未使用的正面圖。檔名格式：

```text
<類別>-<資產ID>-front.<jpg|jpeg|png|webp>
```

支援類別：`character`、`monster`、`equipment`、`prop`、`environment`。例如：

```text
character-novice_female-front.jpg
```

類別名稱必須符合上列拼法；資產 ID 可用英數字、底線或連字號。最好提供去背 PNG；JPG 或不透明圖片會自動嘗試去背。檔名不符合格式，或已以 `used-` 開頭的圖片會被忽略。

## 操作流程

1. 在 Colab 開啟 Notebook，選擇「執行階段 → 變更執行階段類型」：T4 GPU，執行階段版本 `2026.07`。
2. 執行區塊 1 檢查 GPU，接著執行區塊 2 掛載 Drive、建立資料夾並檢查空間。首次授權時依 Colab 提示允許存取 Drive。
3. 將符合命名規則的正面圖放進 `MyDrive/RagnarokChange/hunyuan3d/in/`。
4. 執行區塊 3 安裝環境。新 runtime 安裝完成會自動重新啟動；重啟後依序重跑區塊 1、2、3。之後同一個 runtime 的區塊 3 會快速略過安裝。
5. 執行區塊 4 載入圖片清單，再執行區塊 5 生成並顯示正面、左側、背面、右側參考圖。**此處會停下來等你檢查**，不會自動開始 3D。left 應是主體的左側（角色面朝畫面左方），與 Hunyuan3D-2mv 範例圖的方向一致。
6. 視角不滿意時，在區塊 2 修改 `VIEW_SEED`，重跑區塊 2、4、5 生成另一組；滿意後才手動執行區塊 6。
7. 區塊 6 使用 front/left/back 生成形狀；right 會保留供檢視，但 Hunyuan3D-2mv 不以它作為條件。區塊 7 再產生貼圖。
8. 執行區塊 8，將 GLB 和 metadata 複製到 Drive 的 `out/`，並下載 zip。確認 GLB 已存入 Drive 後，原始正面圖才會加上 `used-` 前綴，避免下次重複處理。

視角圖存在 `in/`，檔名如 `used-character-novice_female-front-cutout.png`（去背後的正面圖，形狀與貼圖都用它）、`...-left.jpg`、`...-back.jpg`、`...-right.jpg`。輸出範例：

```text
out/hunyuan3d2-character-novice_female.glb
out/hunyuan3d2-character-novice_female.json
```

完成後正面來源會變成 `in/used-character-novice_female-front.jpg`。`used-` 開頭的原圖不會再被區塊 4 選取；要重新生成時，請另放一份新的未使用正面圖。

## 可修改設定

所有常用設定集中在區塊 2：

- `DRIVE_ROOT`：Drive 專案根目錄。
- `CACHE_MODELS_ON_DRIVE`：預設 `True`，把 Hugging Face 和去背模型快取放在 Drive。
- `VIEW_COUNT`、`VIEW_STEPS`、`VIEW_SEED`、`VIEW_PROMPT`、`VIEW_NEGATIVE_PROMPT`：視角數量、步數、種子與提示詞。
- `SHAPE_MODEL`、`SHAPE_FLASHVDM`、`PAINT_MODEL`：形狀與貼圖模型版本。
- `PRESETS`：各類別的面數上限與 octree 設定。
- `TEXTURE_SIZE`、`RENDER_SIZE`、`LOW_VRAM`、`SEED`：貼圖解析度、顯存模式及形狀種子。

模型輸出受輸入圖、種子和模型限制；生成的背面或遮擋細節不保證完全符合概念圖，請先檢視多視角，必要時換種子或在 Blender 修整。各模型及權重仍受其各自授權條款限制；使用前請查看對應模型頁與授權。