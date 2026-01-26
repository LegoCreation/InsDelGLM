import logomaker
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import io


class DNALandscape:
    """
    Visualization tool for DNA-BERT with GIF support.
    """

    def __init__(self, model, tokenizer, seq_len=40, font="Arial Rounded MT Bold", use_ic=False):
        self.model = model
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.device = next(model.parameters()).device
        self.font = font
        self.use_ic = use_ic

        # Config
        self.vocab_order = ["A", "C", "G", "T"]
        if "-" in self.tokenizer.get_vocab():
            self.vocab_order.append("-")
        self.dna_ids = [self.tokenizer.tokens_to_ids[n] for n in self.vocab_order]
        self.colors = {
            'A': 'green', 'C': 'blue', 'G': 'orange',
            'T': 'red', '-': 'purple'
        }

    def _compute_landscape(self, sequences):
        """
        Helper: Takes a list of sequences, runs the masked inference loop,
        and returns the probability matrix.
        """
        encoded = self.tokenizer(sequences, return_tensors="pt", padding="max_length", max_length=self.seq_len)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        #landscape_probs = np.zeros((self.seq_len, 5))
        landscape_probs = np.zeros((self.seq_len, len(self.vocab_order)))

        # Inference Loop
        for col_idx in range(self.seq_len):
            target_tensor_col = col_idx + 1
            if target_tensor_col >= input_ids.shape[1]: break

            masked_input = input_ids.clone()
            masked_input[:, target_tensor_col] = self.tokenizer.mask_token_id

            with torch.no_grad():
                outputs = self.model(masked_input, attention_mask=attention_mask, return_dict=True)
                probs = F.softmax(outputs.logits[:, target_tensor_col, :], dim=-1)

            landscape_probs[col_idx] = probs[:, self.dna_ids].cpu().numpy().mean(axis=0)

        return landscape_probs

    def generate_aggregated_gif(self, df_enriched, motif_name, motif_seq, filename=None, min_samples=10, fps=2):
        """
        Generates a GIF showing the landscape changes as the motif moves across positions.
        """
        if filename is None:
            filename = f"{motif_name}_landscape.gif"

        start_col = f"{motif_name}_start"
        if start_col not in df_enriched.columns:
            print(f"Error: Column '{start_col}' missing.")
            return

        position_counts = df_enriched[start_col].value_counts().sort_index()
        valid_starts = [pos for pos in position_counts.index if pos != -1 and position_counts[pos] >= min_samples]

        if not valid_starts:
            print("No positions met the minimum sample criteria.")
            return

        print(f"--- Generating GIF for {motif_name} ({len(valid_starts)} frames) ---")

        frames = []

        for start_pos in valid_starts:
            count = position_counts[start_pos]
            print(f"Rendering Frame: Start Index {start_pos} (N={count})...")

            # 1. Get sequences
            subset = df_enriched[df_enriched[start_col] == start_pos]
            sequences = subset['sequences'].tolist()

            # 2. Compute Landscape (Using Helper)
            landscape_probs = self._compute_landscape(sequences)

            title_str = f"Aggregated: {motif_name} ('{motif_seq}') | Index {start_pos} | N={count}"

            # 3. Capture Plot
            fig = self._render_logo(
                landscape_probs,
                title=title_str,
                vline_start=start_pos,
                motif_len=len(motif_seq),
                show=False
            )

            # 4. Save to RAM buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            frames.append(imageio.imread(buf, format='png'))
            buf.close()
            plt.close(fig)

        print(f"Saving GIF to {filename}...")
        imageio.mimsave(filename, frames, fps=fps, loop=0)
        print("Done!")

    def plot_aggregated(self, df_enriched, motif_name, motif_seq, min_samples=50):
        """
        Generates static plots (restored functionality).
        """
        self.model.eval()

        start_col = f"{motif_name}_start"
        if start_col not in df_enriched.columns:
            print(f"Error: Column '{start_col}' missing.")
            return

        position_counts = df_enriched[start_col].value_counts().sort_index()

        print(f"--- Generating Landscapes for {motif_name} ---")

        for start_pos in position_counts.index:
            if start_pos == -1: continue

            count = position_counts[start_pos]
            if count < min_samples: continue

            print(f"Plotting Start Index {start_pos} (N={count})...")

            subset = df_enriched[df_enriched[start_col] == start_pos]
            sequences = subset['sequences'].tolist()

            # Use Helper Here Too!
            landscape_probs = self._compute_landscape(sequences)

            title_str = f"Aggregated: {motif_name} ('{motif_seq}') | Index {start_pos} | N={count}"

            self._render_logo(
                landscape_probs,
                title=title_str,
                vline_start=start_pos,
                motif_len=len(motif_seq),
                show=True
            )

    def _render_logo(self, probs_matrix, title, vline_start=None, motif_len=0,
                     seq_letters=None, show=True, save=None):
        df_logo = pd.DataFrame(probs_matrix, columns=self.vocab_order)

        # Add information content scaling (use class setting)
        if self.use_ic:
            # Calculate entropy for each position
            entropy = -np.sum(df_logo * np.log2(df_logo + 1e-9), axis=1)
            max_entropy = np.log2(len(self.vocab_order))  # log2(5) for ACGT-
            information_content = max_entropy - entropy

            # Scale probabilities by IC
            df_logo = df_logo.mul(information_content, axis=0)
            ylabel = "Information Content (bits)"
            ylim = [0, max_entropy]
        else:
            ylabel = "Confidence"
            ylim = [0, 1]

        fig, ax = plt.subplots(figsize=(12, 3))
        logo = logomaker.Logo(df_logo,
                              color_scheme=self.colors,
                              shade_below=.5,
                              fade_below=.5,
                              font_name=self.font,
                              ax=ax)

        if vline_start is not None:
            logo.ax.axvline(x=vline_start - 0.5, color='black', linewidth=1.5, linestyle="--")
            logo.ax.axvline(x=vline_start + motif_len - 0.5, color='black', linewidth=1.5, linestyle="--")

        if seq_letters:
            ax2 = logo.ax.twiny()
            ax2.set_xlim(logo.ax.get_xlim())
            ax2.set_xticks(range(len(seq_letters)))
            ax2.set_xticklabels(seq_letters, fontsize=8)
            ax2.tick_params(length=0)

        logo.ax.set_xticks(range(0, len(probs_matrix), 5))
        logo.ax.set_ylim(ylim)
        logo.ax.set_ylabel(ylabel)
        logo.ax.set_title(title, y=1.1 if seq_letters else 1.0)
        logo.ax.set_xlabel("Position ID")

        if save:
            plt.savefig(save, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
            return None
        else:
            return fig

    def plot_single(self, df, target_id, motif_seq=None):
        """
        Method 2: Scans ONE sequence by looking up its ID in the dataframe.
        Automatically grabs 'sequences' and 'labels' columns.

        Args:
            df (pd.DataFrame): The dataframe enriched with 'id', 'sequences', 'labels'.
            target_id (int/str): The value in the 'id' column to search for.
            motif_seq (str): Optional motif string to mark.
        """
        self.model.eval()

        # 1. Lookup Row by ID column
        if 'id' not in df.columns:
            print("Error: DataFrame must have an 'id' column.")
            return

        # Filter for the specific ID
        row_subset = df[df['id'] == target_id]

        if len(row_subset) == 0:
            print(f"Error: ID {target_id} not found in DataFrame.")
            return

        row = row_subset.iloc[0]
        sequence = row['sequences']
        label = row['labels'] if 'labels' in df.columns else "Unknown"

        # Construct Title
        full_title = f"Single Seq | ID: {target_id} | Label: {label}"
        print(f"--- Scanning: {full_title} ---")

        # 2. Tokenize
        encoded = self.tokenizer(sequence, return_tensors="pt", padding="max_length", max_length=self.seq_len)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        #landscape_probs = np.zeros((self.seq_len, 5))
        landscape_probs = np.zeros((self.seq_len, len(self.vocab_order)))

        # 3. Scan
        for col_idx in range(self.seq_len):
            target_tensor_col = col_idx + 1
            if target_tensor_col >= input_ids.shape[1]: break

            masked_input = input_ids.clone()
            masked_input[0, target_tensor_col] = self.tokenizer.mask_token_id

            with torch.no_grad():
                outputs = self.model(masked_input, attention_mask=attention_mask, return_dict=True)
                probs = F.softmax(outputs.logits[0, target_tensor_col], dim=-1)

            landscape_probs[col_idx] = probs[self.dna_ids].cpu().numpy()

        start_idx = sequence.find(motif_seq) if motif_seq else -1

        self._render_logo(
            landscape_probs,
            title=full_title,
            vline_start=start_idx if start_idx != -1 else None,
            motif_len=len(motif_seq) if motif_seq else 0,
            seq_letters=list(sequence)
        )

    def plot_string(self, seq, motif_seq=None, label=None, save=None):
        """
        Method 3: You decide on the sequence to predict on.

        Args:
            seq (str): DNA sequence as string for the model to predict.
            motif_seq (str): Optional motif string to mark.
        """
        self.model.eval()
        
        sequence = seq
        label = label if label else motif_seq

        # Construct Title
        full_title = f"Single Seq | Label: {label}"
        print(f"--- Scanning: {full_title} ---")

        # Tokenize
        encoded = self.tokenizer(sequence, return_tensors="pt", padding="max_length", max_length=self.seq_len)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        landscape_probs = np.zeros((self.seq_len, len(self.vocab_order)))

        # Scan
        for col_idx in range(self.seq_len):
            target_tensor_col = col_idx + 1
            if target_tensor_col >= input_ids.shape[1]: break

            masked_input = input_ids.clone()
            masked_input[0, target_tensor_col] = self.tokenizer.mask_token_id

            with torch.no_grad():
                outputs = self.model(masked_input, attention_mask=attention_mask, return_dict=True)
                probs = F.softmax(outputs.logits[0, target_tensor_col], dim=-1)

            landscape_probs[col_idx] = probs[self.dna_ids].cpu().numpy()

        start_idx = sequence.find(motif_seq) if motif_seq else -1

        self._render_logo(
            landscape_probs,
            title=full_title,
            vline_start=start_idx if start_idx != -1 else None,
            motif_len=len(motif_seq) if motif_seq else 0,
            seq_letters=list(sequence),
            save=save
        )
