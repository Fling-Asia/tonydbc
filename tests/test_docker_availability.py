"""
Test Docker availability for TonyDBC tests

This test verifies that Docker is available and running before other tests
that depend on Docker containers are executed.
"""

import pytest
import docker


class TestDockerAvailability:
    """Test Docker availability for container-based tests"""

    def test_docker_is_available(self):
        """Test that Docker is available and running"""
        try:
            client = docker.from_env()
            client.ping()
            print("Docker is available and running")
        except docker.errors.DockerException as e:
            pytest.fail(
                f"Docker is not available or not running: {e}\n\n"
                "To fix this:\n"
                "1. Install Docker Desktop from https://www.docker.com/products/docker-desktop/\n"
                "2. Start Docker Desktop and wait for it to fully load\n"
                "3. Verify with: docker --version && docker ps\n\n"
                "These tests require Docker to spin up MariaDB containers for testing."
            )
        except Exception as e:
            pytest.fail(f"Unexpected error checking Docker availability: {e}")

    def test_docker_can_pull_mariadb_image(self):
        """Test that Docker can access the MariaDB image we need"""
        try:
            client = docker.from_env()
            # Check if we can access the MariaDB image (this will pull if not present)
            # Use a quick pull check without actually pulling the full image
            try:
                client.images.get("mariadb:10.6")
                print("MariaDB 10.6 image is available locally")
            except docker.errors.ImageNotFound:
                print("MariaDB 10.6 image not found locally, checking if it can be pulled...")
                # Just verify we can access the registry, don't actually pull
                client.api.inspect_distribution("mariadb:10.6")
                print("MariaDB 10.6 image is available from Docker Hub")
        except docker.errors.DockerException as e:
            pytest.fail(
                f"Cannot access MariaDB Docker image: {e}\n\n"
                "This might be due to:\n"
                "1. No internet connection to Docker Hub\n"
                "2. Docker registry access issues\n"
                "3. Docker daemon not running properly"
            )
        except Exception as e:
            pytest.fail(f"Unexpected error checking MariaDB image availability: {e}")

    def test_docker_can_list_containers(self):
        """Test that Docker can list containers (basic API functionality)"""
        try:
            client = docker.from_env()
            # Try to list containers (this is a basic Docker operation)
            containers = client.containers.list(all=True)
            print(f"Docker can list containers successfully ({len(containers)} containers found)")
        except docker.errors.DockerException as e:
            pytest.fail(
                f"Docker cannot list containers: {e}\n\n"
                "This might be due to:\n"
                "1. Insufficient Docker permissions\n"
                "2. Docker daemon issues\n"
                "3. Docker API access problems"
            )
        except Exception as e:
            pytest.fail(f"Unexpected error testing Docker API: {e}")


if __name__ == "__main__":
    # Allow running this test module directly
    pytest.main([__file__, "-v", "-s"])
