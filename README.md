# colab-ImageTo3D

使用 Google Colab T4，從一張正面概念圖生成多視角參考圖，再建立並貼圖 Hunyuan3D GLB。Notebook：[`hunyuan3d2_colab_t4.ipynb`](hunyuan3d2_colab_t4.ipynb)

## 雲端硬碟與資料夾

Notebook 預設使用：

```text
Google Drive / MyDrive / RagnarokChange / hunyuan3d /
├── in/                  # 放正面圖；生成的左右、背面圖也先存在這裡
├── out/                 # 完成的 GLB 與 JSON metadata
├── hf_cache/            # Hugging Face 模型權重快取
└── rembg_cache/         # 去背用 U2Net 權重快取
```

**不需要預先建立這些資料夾。** 第一次執行 Notebook 區塊 2、掛載 Google Drive 後，程式會自動建立它們。之後把正面圖放進 `MyDrive/RagnarokChange/hunyuan3d/in` 即可。若想換位置，先修改區塊 2 的 `DRIVE_ROOT`。

模型權重會在流程首次用到時自動下載到 Drive，不必手動下載或解壓，也不必每次重新下載。區塊 2 會檢查快取及可用空間；估計需要 SDXL 約 24 GB，SD 2.1 約 18 GB，實際用量視模型版本與已快取內容而異。你有 5 TB 空間，足以容納這些模型與輸出。請以 Google Drive 顯示的實際剩餘空間為準。

Colab 的執行環境則不同：Hunyuan3D/MV-Adapter 程式碼、Python 套件和編譯出的 CUDA 擴充安裝在 Colab runtime，不會存在 Drive。runtime 重啟或過期後，可能需要重新安裝；模型權重仍會從 Drive 快取讀取。首次安裝區塊 3 會自動重啟 runtime。

## 會下載哪些模型

只會下載目前設定和執行流程實際使用的模型，不會把下表所有選項全部下載。不同名稱是因為各模型負責不同工作，另外也有品質、速度和顯存取捨。

| 用途 | 預設／可選模型 | 何時載入 |
|---|---|---|
| 正面圖生成左右、背面 | [MV-Adapter](https://huggingface.co/huanngzh/mv-adapter) image-to-multiview adapter；預設 SDXL，亦可選 SD 2.1 | 第一次執行區塊 5 |
| 多視角影像基礎模型 | [SDXL Base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)；或切換成 [Stable Diffusion 2.1 Base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) | 第一次執行區塊 5 |
| SDXL 專用 VAE | [`madebyollin/sdxl-vae-fp16-fix`](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) | 選 SDXL 時，第一次執行區塊 5 |
| 多視角形狀 | [Hunyuan3D-2mv](https://huggingface.co/tencent/Hunyuan3D-2mv) 的 shape checkpoint；預設 `hunyuan3d-dit-v2-mv-turbo`，也可選完整模型 | 第一次執行區塊 6 |
| 產生貼圖 | [Hunyuan3D-2](https://huggingface.co/tencent/Hunyuan3D-2) 的 paint checkpoint；預設完整模型，也可選 turbo | 第一次執行區塊 7 |
| 自動去背 | rembg/U2Net 權重 | 首次處理沒有透明背景的輸入圖時；快取在 `rembg_cache/` |

選項含義：

- `VIEW_QUALITY = "sdxl"`：預設畫質，約需 14 GB GPU VRAM；T4 若 OOM，改為 `"sd21"`，再重跑區塊 2、3、4、5。
- `SHAPE_MODEL` 的完整模型較慢、步數較多；`...-turbo` 較快。兩者都是 Hunyuan 多視角形狀模型。
- `PAINT_MODEL` 的完整模型較慢；`...-turbo` 較快。
- SDXL 與 SD 2.1 的視角 adapter、基礎模型不同，切換品質時會下載所選的另一組權重；已下載檔案仍留在 cache。

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

1. 在 Colab 開啟 Notebook，選擇「執行階段 → 變更執行階段類型 → T4 GPU」。
2. 執行區塊 1 檢查 GPU，接著執行區塊 2 掛載 Drive、建立資料夾並檢查空間。首次授權時依 Colab 提示允許存取 Drive。
3. 將符合命名規則的正面圖放進 `MyDrive/RagnarokChange/hunyuan3d/in/`。
4. 執行區塊 3 安裝環境。第一次完成會自動重新啟動；重啟後依序重跑區塊 1、2、3。之後同一個 runtime 的區塊 3 會快速略過安裝。
5. 執行區塊 4 載入圖片清單，再執行區塊 5 生成並顯示正面、左側、背面、右側參考圖。**此處會停下來等你檢查**，不會自動開始 3D。
6. 視角不滿意時，在區塊 2 修改 `VIEW_SEED`，重跑區塊 2、4、5 生成另一組；滿意後才手動執行區塊 6。
7. 區塊 6 使用 front/left/back 生成形狀；right 會保留供檢視，但 Hunyuan3D-2mv 不以它作為條件。區塊 7 再產生貼圖。
8. 執行區塊 8，將 GLB 和 metadata 複製到 Drive 的 `out/`，並下載 zip。確認 GLB 已存入 Drive 後，原始正面圖才會加上 `used-` 前綴，避免下次重複處理。

視角圖存在 `in/`，檔名如 `used-character-novice_female-left.jpg`、`...-back.jpg`、`...-right.jpg`。輸出範例：

```text
out/hunyuan3d2-character-novice_female.glb
out/hunyuan3d2-character-novice_female.json
```

完成後正面來源會變成 `in/used-character-novice_female-front.jpg`。`used-` 開頭的原圖不會再被區塊 4 選取；要重新生成時，請另放一份新的未使用正面圖。

## 可修改設定

所有常用設定集中在區塊 2：

- `DRIVE_ROOT`：Drive 專案根目錄。
- `CACHE_MODELS_ON_DRIVE`：預設 `True`，把 Hugging Face 和去背模型快取放在 Drive。
- `VIEW_QUALITY`、`VIEW_STEPS`、`VIEW_SEED`、`VIEW_PROMPT`、`VIEW_NEGATIVE_PROMPT`：視角生成模型、品質、種子與提示詞。
- `SHAPE_MODEL`、`PAINT_MODEL`：形狀與貼圖模型版本。
- `PRESETS`：各類別的面數上限與 octree 設定。
- `TEXTURE_SIZE`、`RENDER_SIZE`、`LOW_VRAM`、`SEED`：貼圖解析度、顯存模式及形狀種子。

模型輸出受輸入圖、種子和模型限制；生成的背面或遮擋細節不保證完全符合概念圖，請先檢視多視角，必要時換種子或在 Blender 修整。各模型及權重仍受其各自授權條款限制；使用前請查看對應模型頁與授權。