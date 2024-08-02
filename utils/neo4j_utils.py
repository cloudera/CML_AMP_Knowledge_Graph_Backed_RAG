import os
import time
from kubernetes import client, config
from neo4j import GraphDatabase

config.load_incluster_config()

def get_current_namespace():
    with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace", "r") as f:
        return f.read().strip()
    
def get_parent_pod_name():
    with open("/downward-api/pod.name", "r") as f:
        return f.read().strip()
    
def get_parent_pod_uid():
    with open("/downward-api/pod.uid", "r") as f:
        return f.read().strip()
    
def get_pvc_name_from_parent_pod() -> str:
    pod_name = get_parent_pod_name()
    pod_spec = client.CoreV1Api().read_namespaced_pod(name=pod_name, namespace=get_current_namespace())
    volume_mounts = pod_spec.spec.containers[0].volume_mounts
    for volume_mount in volume_mounts:
        if volume_mount.mount_path == "/home/cdsw":
            return volume_mount.name
    
def get_engine_id():
    return os.getenv("CDSW_ENGINE_ID").strip()

def get_neo4j_credentails():
    return {
        "username": "neo4j",
        "password": "password",
        "uri": f"bolt://{get_neo4j_service_name()}.{get_current_namespace()}:7687",
        "database": "neo4j",
    }

def get_onwer_reference():
    parent_pod_name = get_parent_pod_name()
    parent_pod_uid = get_parent_pod_uid()
    return client.V1OwnerReference(
        api_version="v1",
        kind="Pod",
        name=parent_pod_name,
        uid=parent_pod_uid,
    )

def create_deployment_spec_for_neo4j():
    namespace = get_current_namespace()
    engine_id = get_engine_id()
    
    deployment = client.V1Deployment(
        api_version="apps/v1",
        metadata=client.V1ObjectMeta(
            name=f"neo4j-{engine_id}",
            labels={"app": f"neo4j-{engine_id}"},
            owner_references=[get_onwer_reference()],
            namespace=namespace,
        ),
        spec=client.V1DeploymentSpec(
            replicas=1,
            selector=client.V1LabelSelector(
                match_labels={"app": f"neo4j-{engine_id}"}
            ),
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    labels={"app": f"neo4j-{engine_id}"}
                ),
                spec=client.V1PodSpec(
                    security_context=client.V1PodSecurityContext(
                        fs_group=2000,
                        run_as_group=3000,
                        run_as_user=1000,
                        run_as_non_root=True,
                    ),
                    containers=[
                        client.V1Container(
                            name="neo4j",
                            image="neo4j:5.18.0",
                            ports=[
                                client.V1ContainerPort(container_port=7687, name = "bolt"),
                                client.V1ContainerPort(container_port=7474, name = "http"),
                            ],
                            env=[
                                client.V1EnvVar(name="NEO4J_AUTH", value=f"{get_neo4j_credentails()['username']}/{get_neo4j_credentails()['password']}"),
                                client.V1EnvVar(name="NEO4J_apoc_export_file_enabled", value="true"),
                                client.V1EnvVar(name="NEO4J_apoc_import_file_enabled", value="true"),
                                client.V1EnvVar(name="NEO4J_apoc_import_file_use__neo4j__config", value="true"),
                                client.V1EnvVar(name="NEO4JLABS_PLUGINS", value="[\"apoc\",\"graph-data-science\"]"),
                            ],
                            resources=client.V1ResourceRequirements(
                                requests={"cpu": "1", "memory": "4Gi"},
                                limits={"cpu": "1", "memory": "4Gi"},
                            ),
                            volume_mounts=[
                                client.V1VolumeMount(
                                    name="neo4j-plugins",
                                    mount_path="/plugins",
                                ),
                                client.V1VolumeMount(
                                    name="filesystem-access",
                                    mount_path="/data",
                                    sub_path="neo4j-volume"
                                ),
                            ]
                        )
                    ],
                    volumes=[
                        client.V1Volume(
                            name="neo4j-data",
                            empty_dir=client.V1EmptyDirVolumeSource(),
                        ),
                        client.V1Volume(
                            name="neo4j-plugins",
                            empty_dir=client.V1EmptyDirVolumeSource(),
                        ),
                        client.V1Volume(
                            name="filesystem-access",
                            persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                                claim_name=get_pvc_name_from_parent_pod()
                            ),
                        )
                    ]
                ),
            )
        )
    )
    return deployment

def get_neo4j_service_name():
    return f"cml-neo4j-{get_engine_id()}"

def create_service_spec_for_neo4j():
    namespace = get_current_namespace()
    engine_id = get_engine_id()
    
    service = client.V1Service(
        api_version="v1",
        metadata=client.V1ObjectMeta(
            name=get_neo4j_service_name(),
            labels={"app": f"neo4j-{engine_id}"},
            owner_references=[get_onwer_reference()],
            namespace=namespace,
        ),
        spec=client.V1ServiceSpec(
            selector={"app": f"neo4j-{engine_id}"},
            ports=[
                client.V1ServicePort(port=7687, target_port=7687, name="bolt", protocol="TCP"),
                client.V1ServicePort(port=7474, target_port=7474, name="http", protocol="TCP"),
            ]
        )
    )
    return service

def deploy_neo4j_server(): 
    api_instance = client.AppsV1Api()
    service_api_instance = client.CoreV1Api()
    
    deployment_spec = create_deployment_spec_for_neo4j()
    service_spec = create_service_spec_for_neo4j()
    
    api_instance.create_namespaced_deployment(namespace=get_current_namespace(), body=deployment_spec)
    service_api_instance.create_namespaced_service(namespace=get_current_namespace(), body=service_spec)

def stop_neo4j_server():
    api_instance = client.AppsV1Api()
    service_api_instance = client.CoreV1Api()
    
    engine_id = get_engine_id()
    
    api_instance.delete_namespaced_deployment(
        name=f"neo4j-{engine_id}",
        namespace=get_current_namespace(),
    )
    service_api_instance.delete_namespaced_service(
        name=get_neo4j_service_name(),
        namespace=get_current_namespace(),
    )
    
def reset_neo4j_server():
    try:
        stop_neo4j_server()
    except Exception as e:
        print(f"Failed to stop neo4j server: {e}")
    deploy_neo4j_server()

def is_neo4j_server_up():
    neo4j_auth=(get_neo4j_credentails()["username"], get_neo4j_credentails()["password"])
    with GraphDatabase.driver(get_neo4j_credentails()["uri"], auth=neo4j_auth) as driver:
        try:
            driver.verify_connectivity()
            return True
        except Exception as e:
            return False

def wait_for_neo4j_server(max_retries=10, sleep_duration=10):
    neo4j_auth=(get_neo4j_credentails()["username"], get_neo4j_credentails()["password"])
    with GraphDatabase.driver(get_neo4j_credentails()["uri"], auth=neo4j_auth) as driver:
        for i in range(max_retries):
            try:
               driver.verify_connectivity()
               return
            except Exception as e:
                print(f"Neo4j server is not ready yet. Retrying... {e}")
                time.sleep(sleep_duration)
        raise Exception("Neo4j server is not ready yet. Max retries exceeded.")
