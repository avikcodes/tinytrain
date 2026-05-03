"use client";

import type { DragEvent } from "react";
import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type TaskType = "classification" | "regression";

type EpochPoint = {
  epoch: number;
  train_loss: number;
  val_loss: number;
  val_accuracy: number;
};

type TrainingResults = {
  session_id: string;
  final_accuracy: number;
  final_loss: number;
  training_time: number;
  training_history: EpochPoint[];
  confusion_matrix: number[][];
  model_base64: string;
  task_type: TaskType;
  ai_summary: string;
  progress?: number;
};

type HistoryItem = {
  file_name?: string;
  final_accuracy?: number;
  created_at?: string;
  session_id?: string;
};

const learningRateOptions = [0.001, 0.01, 0.1];
const hiddenSizeOptions = [32, 64, 128];

function formatFileSize(size: number) {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(2)} MB`;
}

function formatDate(dateString?: string) {
  if (!dateString) return "Unknown date";
  const date = new Date(dateString);
  if (Number.isNaN(date.getTime())) return dateString;
  return date.toLocaleString();
}

function chartTooltipFormatter(
  value: number | string | ReadonlyArray<number | string> | undefined,
) {
  if (typeof value === "number") {
    return value.toFixed(4);
  }
  if (Array.isArray(value)) {
    return value.join(", ");
  }
  return value ?? "";
}

function downloadModel(modelBase64: string) {
  const binary = window.atob(modelBase64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  const blob = new Blob([bytes], { type: "application/octet-stream" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "model.pkl";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [targetColumn, setTargetColumn] = useState("");
  const [taskType, setTaskType] = useState<TaskType>("classification");
  const [epochs, setEpochs] = useState(50);
  const [learningRate, setLearningRate] = useState(0.001);
  const [hiddenSize, setHiddenSize] = useState(64);
  const [loading, setLoading] = useState(false);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [totalEpochs, setTotalEpochs] = useState(50);
  const [liveHistory, setLiveHistory] = useState<EpochPoint[]>([]);
  const [results, setResults] = useState<TrainingResults | null>(null);
  const [error, setError] = useState("");
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [progress, setProgress] = useState(0);
  const [progressStep, setProgressStep] = useState("Ready for training");
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const response = await fetch("http://localhost:8000/history");
        if (!response.ok) {
          throw new Error("Failed to fetch history");
        }
        const data = (await response.json()) as HistoryItem[];
        setHistory(Array.isArray(data) ? data : []);
      } catch {
        setHistory([]);
      }
    };

    fetchHistory();

    try {
      const saved = window.localStorage.getItem("tinytrain_last_result");
      if (!saved) return;
      const parsed = JSON.parse(saved) as TrainingResults;
      setResults(parsed);
      setLiveHistory(parsed.training_history ?? []);
      setCurrentEpoch(parsed.training_history?.length ?? 0);
      setTotalEpochs(parsed.training_history?.length ?? epochs);
      setProgress(100);
      setProgressStep("Last training session restored");
    } catch {
      window.localStorage.removeItem("tinytrain_last_result");
    }
  }, []);

  const liveChartData = liveHistory.slice(-50);
  const canStart = Boolean(file && targetColumn && !loading);

  const parseHeaders = (selectedFile: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      const text = String(reader.result ?? "");
      const [firstLine = ""] = text.split(/\r?\n/);
      const parsedColumns = firstLine
        .split(",")
        .map((column) => column.trim().replace(/^"|"$/g, ""))
        .filter(Boolean);
      setColumns(parsedColumns);
      setTargetColumn((previous) =>
        parsedColumns.includes(previous) ? previous : parsedColumns[0] ?? "",
      );
    };
    reader.readAsText(selectedFile);
  };

  const handleFileSelection = (selectedFile: File | null) => {
    setFile(selectedFile);
    setColumns([]);
    setTargetColumn("");
    setResults(null);
    setLiveHistory([]);
    setCurrentEpoch(0);
    setProgress(0);
    setProgressStep("Ready for training");
    setError("");
    if (selectedFile) {
      parseHeaders(selectedFile);
    }
  };

  const handleDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setIsDragging(false);
    const droppedFile = event.dataTransfer.files?.[0] ?? null;
    if (droppedFile && droppedFile.name.toLowerCase().endsWith(".csv")) {
      handleFileSelection(droppedFile);
    } else {
      setError("Please upload a valid CSV file.");
    }
  };

  const startTraining = async () => {
    if (!file || !targetColumn) return;

    setLoading(true);
    setError("");
    setResults(null);
    setLiveHistory([]);
    setCurrentEpoch(0);
    setTotalEpochs(epochs);
    setProgress(0);
    setProgressStep("Preparing dataset...");

    try {
      const fileData = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
          const result = String(reader.result ?? "");
          const base64 = result.includes(",") ? result.split(",")[1] ?? "" : result;
          resolve(base64);
        };
        reader.onerror = () => reject(new Error("Failed to read file."));
        reader.readAsDataURL(file);
      });

      const socket = new WebSocket("ws://localhost:8000/ws/train");

      socket.onopen = () => {
        socket.send(
          JSON.stringify({
            filename: file.name,
            data: fileData,
            target_column: targetColumn,
            epochs,
            learning_rate: learningRate,
            hidden_size: hiddenSize,
            task_type: taskType,
          }),
        );
      };

      socket.onmessage = (event) => {
        const message = JSON.parse(event.data) as
          | {
              type: "epoch";
              epoch: number;
              total_epochs: number;
              train_loss: number;
              val_loss: number;
              val_accuracy: number;
              progress: number;
            }
          | { type: "progress"; step: string; progress: number }
          | ({ type: "complete" } & TrainingResults)
          | { type: "error"; error: string };

        if (message.type === "progress") {
          setProgress(message.progress);
          setProgressStep(message.step);
          return;
        }

        if (message.type === "epoch") {
          setCurrentEpoch(message.epoch);
          setTotalEpochs(message.total_epochs);
          setProgress(message.progress);
          setProgressStep("Training neural network...");
          setLiveHistory((previous) => [
            ...previous,
            {
              epoch: message.epoch,
              train_loss: message.train_loss,
              val_loss: message.val_loss,
              val_accuracy: message.val_accuracy,
            },
          ]);
          return;
        }

        if (message.type === "complete") {
          const completedResults: TrainingResults = {
            session_id: message.session_id,
            final_accuracy: message.final_accuracy,
            final_loss: message.final_loss,
            training_time: message.training_time,
            training_history: message.training_history,
            confusion_matrix: message.confusion_matrix,
            model_base64: message.model_base64,
            task_type: message.task_type,
            ai_summary: message.ai_summary,
            progress: message.progress,
          };
          setResults(completedResults);
          setLiveHistory(message.training_history);
          setCurrentEpoch(message.training_history.length);
          setTotalEpochs(message.training_history.length);
          setProgress(100);
          setProgressStep("Training complete");
          setLoading(false);
          window.localStorage.setItem(
            "tinytrain_last_result",
            JSON.stringify(completedResults),
          );
          socket.close();
          return;
        }

        if (message.type === "error") {
          setError(message.error);
          setLoading(false);
          setProgressStep("Training failed");
          socket.close();
        }
      };

      socket.onerror = () => {
        setError("Unable to connect to the training server.");
        setLoading(false);
        setProgressStep("Connection error");
      };

      socket.onclose = () => {
        setLoading(false);
      };
    } catch (trainingError) {
      setError(
        trainingError instanceof Error
          ? trainingError.message
          : "Something went wrong while starting training.",
      );
      setLoading(false);
      setProgressStep("Startup failed");
    }
  };

  return (
    <main className="min-h-screen bg-[#0a0a0a] text-white">
      <div className="mx-auto flex min-h-screen max-w-7xl flex-col lg:flex-row">
        <aside className="border-b border-white/10 bg-[#0f0f0f] px-5 py-6 lg:w-80 lg:border-r lg:border-b-0 lg:px-6">
          <div className="sticky top-0 space-y-6">
            <div>
              <p className="text-xs uppercase tracking-[0.28em] text-[#10b981]">
                Session History
              </p>
              <h2 className="mt-3 text-2xl font-semibold text-white">Recent Runs</h2>
              <p className="mt-2 text-sm leading-6 text-white/55">
                The last ten training sessions pulled from your FastAPI backend.
              </p>
            </div>

            <div className="space-y-3">
              {history.length === 0 ? (
                <div className="rounded-3xl border border-dashed border-white/10 bg-white/[0.02] p-5 text-sm text-white/50">
                  No past training sessions yet
                </div>
              ) : (
                history.map((item, index) => (
                  <div
                    key={item.session_id ?? `${item.file_name ?? "history"}-${index}`}
                    className="rounded-3xl border border-white/8 bg-white/[0.03] p-4 shadow-[0_0_0_1px_rgba(255,255,255,0.02)]"
                  >
                    <p className="truncate text-sm font-medium text-white">
                      {item.file_name ?? "Untitled dataset"}
                    </p>
                    <div className="mt-3 flex items-center justify-between text-xs text-white/50">
                      <span>
                        Acc.{" "}
                        {typeof item.final_accuracy === "number"
                          ? item.final_accuracy.toFixed(4)
                          : "--"}
                      </span>
                      <span>{formatDate(item.created_at)}</span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </aside>

        <section className="flex-1 px-5 py-6 sm:px-8 lg:px-10 lg:py-8">
          <div className="mx-auto max-w-5xl space-y-8">
            <header className="rounded-[2rem] border border-white/10 bg-[radial-gradient(circle_at_top_left,rgba(16,185,129,0.15),transparent_35%),linear-gradient(180deg,rgba(255,255,255,0.03),rgba(255,255,255,0.01))] p-8 shadow-[0_30px_120px_rgba(0,0,0,0.35)]">
              <span className="inline-flex items-center rounded-full border border-[#10b981]/30 bg-[#10b981]/10 px-3 py-1 text-xs font-medium tracking-[0.2em] text-[#86efac] uppercase">
                Open Source • Free Forever
              </span>
              <h1 className="mt-5 text-4xl font-bold tracking-tight text-white sm:text-6xl">
                TinyTrain
              </h1>
              <p className="mt-4 max-w-2xl text-lg text-white/80 sm:text-xl">
                Upload any CSV. Train a neural network. Watch it learn.
              </p>
              <p className="mt-3 max-w-2xl text-sm leading-7 text-white/55 sm:text-base">
                Fine-tunes a small neural network on your data and streams live
                training loss.
              </p>
            </header>

            <section className="rounded-[2rem] border border-white/10 bg-[#111111] p-6 shadow-[0_20px_80px_rgba(0,0,0,0.28)] sm:p-8">
              <div className="grid gap-8 xl:grid-cols-[1.3fr_0.9fr]">
                <div className="space-y-6">
                  <div>
                    <p className="text-sm uppercase tracking-[0.24em] text-white/40">
                      Upload Dataset
                    </p>
                    <label
                      onDragOver={(event) => {
                        event.preventDefault();
                        setIsDragging(true);
                      }}
                      onDragLeave={() => setIsDragging(false)}
                      onDrop={handleDrop}
                      className={`mt-4 flex min-h-[240px] cursor-pointer flex-col items-center justify-center rounded-[1.75rem] border border-dashed px-6 text-center transition ${
                        isDragging
                          ? "border-[#10b981] bg-[#10b981]/10"
                          : "border-white/15 bg-[#0d0d0d] hover:border-[#10b981]/60 hover:bg-[#121212]"
                      }`}
                    >
                      <input
                        type="file"
                        accept=".csv"
                        className="hidden"
                        onChange={(event) =>
                          handleFileSelection(event.target.files?.[0] ?? null)
                        }
                      />
                      <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-[#10b981]/25 bg-[#10b981]/10 text-2xl text-[#10b981]">
                        CSV
                      </div>
                      <p className="mt-5 text-lg font-medium text-white">
                        Drag and drop your CSV here
                      </p>
                      <p className="mt-2 text-sm text-white/45">
                        or click to browse. TinyTrain parses headers locally before
                        anything gets sent.
                      </p>
                      {file ? (
                        <div className="mt-6 rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-3 text-sm text-white/75">
                          {file.name} • {formatFileSize(file.size)}
                        </div>
                      ) : null}
                    </label>
                  </div>

                  {file ? (
                    <div className="grid gap-4 sm:grid-cols-2">
                      <div>
                        <label className="mb-2 block text-sm text-white/65">
                          Target Column
                        </label>
                        <select
                          value={targetColumn}
                          onChange={(event) => setTargetColumn(event.target.value)}
                          className="w-full rounded-2xl border border-white/10 bg-[#0d0d0d] px-4 py-3 text-sm text-white outline-none transition focus:border-[#10b981]/70"
                        >
                          {columns.map((column) => (
                            <option key={column} value={column}>
                              {column}
                            </option>
                          ))}
                        </select>
                      </div>

                      <div>
                        <label className="mb-2 block text-sm text-white/65">
                          Task Type
                        </label>
                        <div className="grid grid-cols-2 gap-3">
                          {(["classification", "regression"] as TaskType[]).map((type) => (
                            <button
                              key={type}
                              type="button"
                              onClick={() => setTaskType(type)}
                              className={`rounded-2xl border px-4 py-3 text-sm font-medium capitalize transition ${
                                taskType === type
                                  ? "border-[#10b981] bg-[#10b981] text-[#052e1f]"
                                  : "border-white/10 bg-[#0d0d0d] text-white/70 hover:border-white/20 hover:text-white"
                              }`}
                            >
                              {type}
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>
                  ) : null}
                </div>

                <div className="space-y-6 rounded-[1.75rem] border border-white/8 bg-[#0d0d0d] p-5">
                  <div>
                    <p className="text-sm uppercase tracking-[0.24em] text-white/40">
                      Training Config
                    </p>
                    <div className="mt-5 space-y-5">
                      <div>
                        <div className="mb-2 flex items-center justify-between text-sm text-white/65">
                          <span>Epochs</span>
                          <span className="font-medium text-[#86efac]">{epochs}</span>
                        </div>
                        <input
                          type="range"
                          min={10}
                          max={100}
                          value={epochs}
                          onChange={(event) => setEpochs(Number(event.target.value))}
                          className="h-2 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-[#10b981]"
                        />
                      </div>

                      <div className="grid gap-4 sm:grid-cols-2">
                        <div>
                          <label className="mb-2 block text-sm text-white/65">
                            Learning Rate
                          </label>
                          <select
                            value={learningRate}
                            onChange={(event) =>
                              setLearningRate(Number(event.target.value))
                            }
                            className="w-full rounded-2xl border border-white/10 bg-[#101010] px-4 py-3 text-sm text-white outline-none transition focus:border-[#10b981]/70"
                          >
                            {learningRateOptions.map((option) => (
                              <option key={option} value={option}>
                                {option}
                              </option>
                            ))}
                          </select>
                        </div>

                        <div>
                          <label className="mb-2 block text-sm text-white/65">
                            Hidden Size
                          </label>
                          <select
                            value={hiddenSize}
                            onChange={(event) =>
                              setHiddenSize(Number(event.target.value))
                            }
                            className="w-full rounded-2xl border border-white/10 bg-[#101010] px-4 py-3 text-sm text-white outline-none transition focus:border-[#10b981]/70"
                          >
                            {hiddenSizeOptions.map((option) => (
                              <option key={option} value={option}>
                                {option}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="rounded-2xl border border-white/8 bg-white/[0.02] p-4">
                    <p className="text-sm text-white/45">Status</p>
                    <p className="mt-2 text-base text-white">{progressStep}</p>
                  </div>

                  <button
                    type="button"
                    onClick={startTraining}
                    disabled={!canStart}
                    className="w-full rounded-2xl bg-[#10b981] px-5 py-4 text-sm font-semibold text-[#06281c] transition hover:bg-[#34d399] disabled:cursor-not-allowed disabled:bg-[#184636] disabled:text-white/35"
                  >
                    {loading ? "Training..." : "Start Training"}
                  </button>
                </div>
              </div>
            </section>

            {error ? (
              <section className="rounded-[1.75rem] border border-red-500/30 bg-red-500/10 p-5 text-sm text-red-100">
                {error}
              </section>
            ) : null}

            {(loading || liveHistory.length > 0) && !results ? (
              <section className="rounded-[2rem] border border-white/10 bg-[#111111] p-6 sm:p-8">
                <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
                  <div>
                    <p className="text-sm uppercase tracking-[0.22em] text-white/40">
                      Live Training
                    </p>
                    <h2 className="mt-3 text-3xl font-semibold text-white">
                      Epoch {currentEpoch} / {totalEpochs}
                    </h2>
                  </div>
                  <div className="inline-flex w-fit items-center rounded-full border border-[#10b981]/30 bg-[#10b981]/10 px-4 py-2 text-sm font-medium text-[#86efac]">
                    Accuracy{" "}
                    {liveHistory.length > 0
                      ? liveHistory[liveHistory.length - 1]?.val_accuracy.toFixed(4)
                      : "0.0000"}
                  </div>
                </div>

                <div className="mt-8 h-80 rounded-[1.5rem] border border-white/8 bg-[#0c0c0c] p-4">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={liveChartData}>
                      <CartesianGrid stroke="rgba(255,255,255,0.08)" vertical={false} />
                      <XAxis dataKey="epoch" stroke="rgba(255,255,255,0.35)" />
                      <YAxis stroke="rgba(255,255,255,0.35)" />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "#111111",
                          border: "1px solid rgba(255,255,255,0.08)",
                          borderRadius: "16px",
                          color: "#ffffff",
                        }}
                        formatter={chartTooltipFormatter}
                      />
                      <Line
                        type="monotone"
                        dataKey="train_loss"
                        stroke="#10b981"
                        strokeWidth={3}
                        dot={false}
                      />
                      <Line
                        type="monotone"
                        dataKey="val_loss"
                        stroke="#f59e0b"
                        strokeWidth={3}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                <div className="mt-6">
                  <div className="mb-2 flex items-center justify-between text-xs uppercase tracking-[0.2em] text-white/35">
                    <span>Progress</span>
                    <span>{progress}%</span>
                  </div>
                  <div className="h-3 rounded-full bg-white/8">
                    <div
                      className="h-3 rounded-full bg-[#10b981] transition-all duration-300"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>
              </section>
            ) : null}

            {results ? (
              <section className="space-y-8">
                <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                  {[
                    {
                      label: "Final Accuracy",
                      value: results.final_accuracy.toFixed(4),
                    },
                    {
                      label: "Final Loss",
                      value: results.final_loss.toFixed(4),
                    },
                    {
                      label: "Total Epochs",
                      value: String(results.training_history.length),
                    },
                    {
                      label: "Training Time",
                      value: `${results.training_time}s`,
                    },
                  ].map((stat) => (
                    <div
                      key={stat.label}
                      className="rounded-[1.75rem] border border-white/10 bg-[#111111] p-5"
                    >
                      <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-[#10b981]/12 text-[#10b981]">
                        ●
                      </div>
                      <p className="mt-5 text-3xl font-semibold text-white">{stat.value}</p>
                      <p className="mt-2 text-sm text-white/45">{stat.label}</p>
                    </div>
                  ))}
                </div>

                <div className="rounded-[2rem] border border-white/10 bg-[#111111] p-6 sm:p-8">
                  <h3 className="text-2xl font-semibold text-white">
                    Training Loss Curve
                  </h3>
                  <div className="mt-6 h-96 rounded-[1.5rem] border border-white/8 bg-[#0c0c0c] p-4">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={results.training_history}>
                        <CartesianGrid
                          stroke="rgba(255,255,255,0.08)"
                          vertical={false}
                        />
                        <XAxis dataKey="epoch" stroke="rgba(255,255,255,0.35)" />
                        <YAxis stroke="rgba(255,255,255,0.35)" />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: "#111111",
                            border: "1px solid rgba(255,255,255,0.08)",
                            borderRadius: "16px",
                            color: "#ffffff",
                          }}
                          formatter={chartTooltipFormatter}
                        />
                        <Line
                          type="monotone"
                          dataKey="train_loss"
                          stroke="#10b981"
                          strokeWidth={3}
                          dot={false}
                        />
                        <Line
                          type="monotone"
                          dataKey="val_loss"
                          stroke="#f59e0b"
                          strokeWidth={3}
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {results.task_type === "classification" &&
                results.confusion_matrix.length > 0 ? (
                  <div className="rounded-[2rem] border border-white/10 bg-[#111111] p-6 sm:p-8">
                    <h3 className="text-2xl font-semibold text-white">
                      Confusion Matrix
                    </h3>
                    <div className="mt-6 overflow-x-auto">
                      <div
                        className="grid gap-2"
                        style={{
                          gridTemplateColumns: `repeat(${
                            results.confusion_matrix[0]?.length + 1
                          }, minmax(72px, 1fr))`,
                        }}
                      >
                        <div className="flex items-center justify-center rounded-2xl bg-transparent text-xs text-white/30">
                          Actual / Pred
                        </div>
                        {results.confusion_matrix[0]?.map((_, columnIndex) => (
                          <div
                            key={`column-${columnIndex}`}
                            className="flex items-center justify-center rounded-2xl bg-white/[0.04] px-3 py-4 text-sm text-white/60"
                          >
                            {columnIndex}
                          </div>
                        ))}
                        {results.confusion_matrix.map((row, rowIndex) => (
                          <FragmentRow
                            key={`row-${rowIndex}`}
                            row={row}
                            rowIndex={rowIndex}
                          />
                        ))}
                      </div>
                    </div>
                  </div>
                ) : null}

                <div className="rounded-[2rem] border border-white/10 bg-[#111111] p-6 sm:p-8">
                  <div className="border-l-4 border-[#10b981] pl-5">
                    <h3 className="text-2xl font-semibold text-white">
                      Training Summary
                    </h3>
                    <p className="mt-4 max-w-3xl text-base leading-8 text-white/72">
                      {results.ai_summary}
                    </p>
                  </div>
                </div>

                <button
                  type="button"
                  onClick={() => downloadModel(results.model_base64)}
                  className="w-full rounded-[1.75rem] bg-[#10b981] px-6 py-5 text-base font-semibold text-[#06281c] transition hover:bg-[#34d399]"
                >
                  Download Trained Model (.pkl)
                </button>
              </section>
            ) : null}

            <footer className="border-t border-white/8 pt-2 text-center text-sm text-white/40">
              Built by @avikcodes • Project 7 of 30
            </footer>
          </div>
        </section>
      </div>
    </main>
  );
}

function FragmentRow({
  row,
  rowIndex,
}: {
  row: number[];
  rowIndex: number;
}) {
  const maxValue = Math.max(...row, 1);

  return (
    <>
      <div className="flex items-center justify-center rounded-2xl bg-white/[0.04] px-3 py-4 text-sm text-white/60">
        {rowIndex}
      </div>
      {row.map((value, columnIndex) => {
        const intensity = value / maxValue;
        const isDiagonal = rowIndex === columnIndex;
        const background = isDiagonal
          ? `rgba(16, 185, 129, ${0.18 + intensity * 0.55})`
          : `rgba(255, 255, 255, ${0.04 + intensity * 0.08})`;

        return (
          <div
            key={`${rowIndex}-${columnIndex}`}
            className="flex items-center justify-center rounded-2xl border border-white/6 px-3 py-4 text-sm font-medium text-white"
            style={{ backgroundColor: background }}
          >
            {value}
          </div>
        );
      })}
    </>
  );
}
