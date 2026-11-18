import { useEffect, useRef, useState } from "react";

const ACCEPTED_IMAGES = ".png,.jpg,.jpeg,.bmp,.tif,.tiff";

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function downloadCsv(results) {
  const columns = ["Image", "Predicted character", "Confidence", "Status"];
  const escapeCell = (value) => `"${String(value ?? "").replaceAll('"', '""')}"`;
  const rows = results.map((item) =>
    columns
      .map((column) =>
        escapeCell(
          column === "Confidence" && item[column] !== null
            ? `${(item[column] * 100).toFixed(2)}%`
            : item[column],
        ),
      )
      .join(","),
  );
  const csv = `\uFEFF${[columns.map(escapeCell).join(","), ...rows].join("\r\n")}`;
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = "character_predictions.csv";
  link.click();
  URL.revokeObjectURL(url);
}

function App() {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [alreadyPreprocessed, setAlreadyPreprocessed] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const isZip = file?.name.toLowerCase().endsWith(".zip") ?? false;
  const [imageUrl, setImageUrl] = useState("");

  useEffect(() => {
    if (!file || isZip) {
      setImageUrl("");
      return undefined;
    }
    const url = URL.createObjectURL(file);
    setImageUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file, isZip]);

  function chooseFile(nextFile) {
    if (!nextFile) return;
    const name = nextFile.name.toLowerCase();
    if (!name.endsWith(".zip") && !ACCEPTED_IMAGES.split(",").some((ext) => name.endsWith(ext))) {
      setError("Choose a supported image or ZIP archive.");
      return;
    }
    setFile(nextFile);
    setResult(null);
    setError("");
  }

  async function submit(event) {
    event.preventDefault();
    if (!file || loading) return;

    setLoading(true);
    setError("");
    setResult(null);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("already_preprocessed", String(alreadyPreprocessed));

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Prediction failed.");
      setResult(data);
    } catch (requestError) {
      setError(requestError.message || "Could not reach the prediction service.");
    } finally {
      setLoading(false);
    }
  }

  const zipResults = result?.type === "zip" ? result.results : [];
  const successfulCount = zipResults.filter((item) => item.Status === "OK").length;

  return (
    <main className="page-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="Malayalam OCR home">
          <span className="brand-mark" aria-hidden="true">അ</span>
          <span>akshara<span className="brand-light">.lab</span></span>
        </a>
        <span className="topbar-note"><span className="status-dot" /> Malayalam character recognition</span>
      </header>

      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow"><span className="eyebrow-line" /> HANDWRITING, UNDERSTOOD</p>
          <h1>Every character<br />has a <span>story.</span></h1>
          <p className="hero-description">
            Turn handwritten Malayalam into text. Upload a character to identify it,
            or a ZIP of images to process them all at once.
          </p>
          <div className="hero-meta">
            <span><span className="meta-icon">01</span> Single image recognition</span>
            <span><span className="meta-icon">02</span> Batch ZIP processing</span>
          </div>
        </div>
        <div className="hero-script" aria-hidden="true">
          <span>മലയാളം</span>
          <i>അക്ഷരങ്ങൾ</i>
        </div>
      </section>

      <section className="workspace" aria-label="Character recognition">
        <div className="upload-panel">
          <div className="panel-heading">
            <div>
              <p className="section-index">01 / UPLOAD</p>
              <h2>Bring your character</h2>
            </div>
            <span className="format-badge">IMAGE OR ZIP</span>
          </div>

          <form onSubmit={submit}>
            <input
              ref={inputRef}
              className="visually-hidden"
              type="file"
              accept={`${ACCEPTED_IMAGES},.zip`}
              onChange={(event) => chooseFile(event.target.files?.[0])}
            />
            <button
              className={`drop-zone${dragging ? " is-dragging" : ""}${file ? " has-file" : ""}`}
              type="button"
              onClick={() => inputRef.current?.click()}
              onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => {
                event.preventDefault();
                setDragging(false);
                chooseFile(event.dataTransfer.files?.[0]);
              }}
              aria-label="Choose or drop an image or ZIP file"
            >
              <span className="upload-icon" aria-hidden="true">
                {isZip ? "ZIP" : <svg viewBox="0 0 24 24" fill="none"><path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M5 14v5h14v-5" /></svg>}
              </span>
              {file ? (
                <>
                  <strong className="file-name">{file.name}</strong>
                  <span>{formatSize(file.size)} <span className="muted-separator">·</span> Click to replace</span>
                </>
              ) : (
                <>
                  <strong>Drop your file here</strong>
                  <span>or <span className="browse-link">browse files</span></span>
                  <small>PNG, JPG, TIFF, BMP or ZIP</small>
                </>
              )}
            </button>

            <label className="preprocess-option">
              <input
                type="checkbox"
                checked={alreadyPreprocessed}
                onChange={(event) => setAlreadyPreprocessed(event.target.checked)}
              />
              <span className="custom-check" aria-hidden="true" />
              <span>
                <strong>Already preprocessed</strong>
                <small>White character on a black background, like training images.</small>
              </span>
              <span className="info-mark" title="Skips automatic polarity correction and cropping.">i</span>
            </label>

            {error && <p className="error-message" role="alert">{error}</p>}

            <button className="submit-button" type="submit" disabled={!file || loading}>
              {loading ? (
                <><span className="spinner" /> Reading your character…</>
              ) : (
                <>Recognize character <span aria-hidden="true">↗</span></>
              )}
            </button>
            <p className="privacy-note"><span aria-hidden="true">◇</span> Your images are processed securely and not stored.</p>
          </form>
        </div>

        <div className="result-panel">
          <div className="panel-heading">
            <div>
              <p className="section-index">02 / RESULT</p>
              <h2>{result ? "Recognition complete" : "Your result, revealed"}</h2>
            </div>
            {result && <span className="complete-badge"><span /> COMPLETE</span>}
          </div>

          {!result && !file && (
            <div className="empty-result">
              <div className="empty-art" aria-hidden="true">
                <span className="art-orbit orbit-one" />
                <span className="art-orbit orbit-two" />
                <span className="art-character">ക</span>
                <span className="art-spark spark-one">✳</span>
                <span className="art-spark spark-two">✳</span>
              </div>
              <p>Your prediction will appear here</p>
              <span>Upload an image to get started</span>
            </div>
          )}

          {!result && file && !isZip && (
            <div className="image-preview-card">
              <img src={imageUrl} alt="Selected character preview" />
              <div><span className="preview-dot" /> Ready to recognize <strong>{file.name}</strong></div>
            </div>
          )}

          {!result && isZip && (
            <div className="zip-ready">
              <div className="zip-icon">ZIP</div>
              <strong>Batch ready to process</strong>
              <span>{file.name} <i>·</i> {formatSize(file.size)}</span>
              <small>Results for each supported image will appear here.</small>
            </div>
          )}

          {result?.type === "image" && (
            <div className="image-result">
              <div className="character-card">
                <div className="recognized-character">{result.predictions[0]?.character}</div>
                <span>TOP MATCH</span>
              </div>
              <div className="prediction-summary">
                <p className="result-label">PREDICTED CHARACTER</p>
                <h3>{result.predictions[0]?.character}</h3>
                <p className="confidence-line">
                  {((result.predictions[0]?.confidence || 0) * 100).toFixed(1)}% confidence
                </p>
                <div className="prediction-list">
                  {result.predictions.map((prediction, index) => (
                    <div className="prediction-row" key={`${prediction.character}-${index}`}>
                      <span className="prediction-rank">0{index + 1}</span>
                      <span className="prediction-character">{prediction.character}</span>
                      <span className="prediction-track"><i style={{ width: `${prediction.confidence * 100}%` }} /></span>
                      <span className="prediction-percent">{(prediction.confidence * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="processed-preview">
                <img src={result.preview} alt="Processed 64 by 64 model input" />
                <span>MODEL INPUT · 64 × 64</span>
              </div>
              <p className="preprocessing-caption">{result.preprocessing_mode}</p>
            </div>
          )}

          {result?.type === "zip" && (
            <div className="batch-result">
              <div className="batch-summary">
                <div><strong>{successfulCount}</strong><span>images recognized</span></div>
                <div><strong>{zipResults.length}</strong><span>total files</span></div>
                <button type="button" className="download-button" onClick={() => downloadCsv(zipResults)}>
                  <span aria-hidden="true">↓</span> Download CSV
                </button>
              </div>
              <div className="table-wrap">
                <table>
                  <thead><tr><th>IMAGE</th><th>CHARACTER</th><th>CONFIDENCE</th><th>STATUS</th></tr></thead>
                  <tbody>
                    {zipResults.map((item, index) => (
                      <tr key={`${item.Image}-${index}`}>
                        <td className="table-filename" title={item.Image}>{item.Image}</td>
                        <td className="table-character">{item["Predicted character"] || "—"}</td>
                        <td>{item.Confidence === null ? "—" : `${(item.Confidence * 100).toFixed(1)}%`}</td>
                        <td><span className={`table-status${item.Status === "OK" ? " status-ok" : " status-error"}`}>{item.Status === "OK" ? "Ready" : "Error"}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <div className="result-footer">
            <span><i /> MODEL STATUS: ONLINE</span>
            <span>POWERED BY CNN + KAN</span>
          </div>
        </div>
      </section>

      <footer className="page-footer">
        <span>BUILT FOR THE BEAUTY OF MALAYALAM</span>
        <span>അ · മലയാളം OCR</span>
      </footer>
    </main>
  );
}

export default App;
