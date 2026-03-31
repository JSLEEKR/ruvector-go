// Command ruvector starts the vector database server.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/JSLEEKR/ruvector-go/pkg/collection"
	"github.com/JSLEEKR/ruvector-go/pkg/server"
)

var (
	version = "1.0.0"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	fs := flag.NewFlagSet("ruvector", flag.ExitOnError)
	addr := fs.String("addr", ":8080", "listen address")
	dataDir := fs.String("data", "./data", "data directory")
	showVersion := fs.Bool("version", false, "show version")

	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("parse flags: %w", err)
	}

	if *showVersion {
		fmt.Printf("ruvector-go v%s\n", version)
		return nil
	}

	manager, err := collection.NewManager(*dataDir)
	if err != nil {
		return fmt.Errorf("create manager: %w", err)
	}

	srv := server.New(manager)

	// Graceful shutdown
	_, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigCh
		log.Println("shutting down...")
		if err := manager.SaveAll(); err != nil {
			log.Printf("save error: %v", err)
		}
		shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer shutdownCancel()
		srv.Shutdown(shutdownCtx)
		cancel()
	}()

	log.Printf("ruvector-go v%s starting on %s (data: %s)", version, *addr, *dataDir)
	if err := srv.Start(*addr); err != nil && err.Error() != "http: Server closed" {
		return fmt.Errorf("server: %w", err)
	}

	return nil
}
