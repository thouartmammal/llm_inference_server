package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sync"
)

type pythonTokenizer struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	input  *json.Encoder
	output *bufio.Scanner
	mu     sync.Mutex
}

type tokenizerCommand struct {
	Operation string `json:"operation"`
	Prompt    string `json:"prompt,omitempty"`
	TokenIDs  []int  `json:"token_ids,omitempty"`
}

type tokenizerResult struct {
	Ready    bool    `json:"ready,omitempty"`
	TokenIDs []int32 `json:"token_ids,omitempty"`
	Text     string  `json:"text,omitempty"`
	Error    string  `json:"error,omitempty"`
}

func startPythonTokenizer(pythonExecutable, helperPath, modelName string) (*pythonTokenizer, error) {
	cmd := exec.Command(pythonExecutable, "-u", helperPath, modelName)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("create tokenizer stdin: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("create tokenizer stdout: %w", err)
	}
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start Python tokenizer process: %w", err)
	}

	output := bufio.NewScanner(stdout)
	output.Buffer(make([]byte, 4096), 16<<20)
	if !output.Scan() {
		scanErr := output.Err()
		waitErr := cmd.Wait()
		if scanErr != nil {
			return nil, fmt.Errorf("read tokenizer startup response: %w", scanErr)
		}
		if waitErr != nil {
			return nil, fmt.Errorf("tokenizer helper exited before ready: %w", waitErr)
		}
		return nil, fmt.Errorf("tokenizer helper closed stdout before becoming ready")
	}
	var result tokenizerResult
	if err := json.Unmarshal(output.Bytes(), &result); err != nil {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		return nil, fmt.Errorf("read tokenizer startup response: %w", err)
	}
	if result.Error != "" {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		return nil, fmt.Errorf("tokenizer helper startup: %s", result.Error)
	}
	if !result.Ready {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		return nil, fmt.Errorf("tokenizer helper returned no ready signal")
	}

	return &pythonTokenizer{
		cmd:    cmd,
		stdin:  stdin,
		input:  json.NewEncoder(stdin),
		output: output,
	}, nil
}

func (p *pythonTokenizer) EncodePrompt(prompt string) ([]int32, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	result, err := p.call(tokenizerCommand{
		Operation: "encode",
		Prompt:    prompt,
	})
	if err != nil {
		return nil, err
	}
	if len(result.TokenIDs) == 0 {
		return nil, fmt.Errorf("tokenizer returned no prompt token IDs")
	}
	return result.TokenIDs, nil
}

func (p *pythonTokenizer) Decode(tokenIDs []int) (string, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	result, err := p.call(tokenizerCommand{
		Operation: "decode",
		TokenIDs:  tokenIDs,
	})
	if err != nil {
		return "", err
	}
	return result.Text, nil
}

func (p *pythonTokenizer) call(command tokenizerCommand) (tokenizerResult, error) {
	if err := p.input.Encode(command); err != nil {
		return tokenizerResult{}, fmt.Errorf("write tokenizer command: %w", err)
	}
	if !p.output.Scan() {
		if err := p.output.Err(); err != nil {
			return tokenizerResult{}, fmt.Errorf("read tokenizer response: %w", err)
		}
		return tokenizerResult{}, fmt.Errorf("tokenizer helper exited unexpectedly")
	}
	var result tokenizerResult
	if err := json.Unmarshal(p.output.Bytes(), &result); err != nil {
		return tokenizerResult{}, fmt.Errorf("parse tokenizer response: %w", err)
	}
	if result.Error != "" {
		return tokenizerResult{}, fmt.Errorf("tokenizer operation %q: %s", command.Operation, result.Error)
	}
	return result, nil
}

func (p *pythonTokenizer) Close() error {
	if err := p.stdin.Close(); err != nil {
		_ = p.cmd.Process.Kill()
		_ = p.cmd.Wait()
		return fmt.Errorf("close tokenizer helper input: %w", err)
	}
	if err := p.cmd.Wait(); err != nil {
		return fmt.Errorf("wait for tokenizer helper: %w", err)
	}
	return nil
}
