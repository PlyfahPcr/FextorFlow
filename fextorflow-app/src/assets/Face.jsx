import * as THREE from 'three'
import React, { useRef } from 'react'
import { useGLTF } from '@react-three/drei'
import { useFrame } from '@react-three/fiber'

export function Face({ emotion = "neutral" }) {
  const group = useRef()
  const { nodes, materials } = useGLTF('/FACE/Face.gltf')

  const emotionColorMap = {
    happy: '#00ff88',
    sad: '#4477ff',
    angry: '#ff3344',
    surprise: '#ffcc00',
    disgust: '#996600',
    fear: '#8844ff', 
    neutral: '#aaaaaa',
  }

  const faceColor = new THREE.Color(emotionColorMap[emotion] || '#ffffff')

  // clone material เพื่อไม่กระทบ global material
  const faceMaterial = materials.Base.clone()
  faceMaterial.color = faceColor

  return (
    <group ref={group} dispose={null} scale={0.9}>
      {/* โครงหน้า */}
      <group position={[0.012, 1.72, 0.325]}>
        <mesh geometry={nodes.Cube_1.geometry} material={faceMaterial} />
        <mesh geometry={nodes.Cube_2.geometry} material={nodes.Cube_2.material} />
      </group>

      {/* ดวงตา */}
      <group position={[0.308, 1.663, 0.829]} scale={0.191}>
        <mesh geometry={nodes.Sphere_1.geometry} material={materials['Eye Ball']} />
        <mesh geometry={nodes.Sphere_2.geometry} material={materials.pupl} />
      </group>

      {emotion === 'happy' && (
        <mesh
          geometry={nodes.Cube045.geometry}
          material={materials.angry_mounth}
          position={[0.087, 1.32, 0.922]}
          rotation={[0, 0, 2.8]}
          scale={[0.25, 0.12, 0.1]}
        />
      )}

      {emotion === 'angry' && (
        <mesh
          geometry={nodes.Cube045.geometry}
          material={materials.angry_mounth}
          position={[0.087, 1.32, 0.922]}
          rotation={[0, 0, 3.2]}
          scale={[0.2, 0.08, 0.1]}
        />
      )}

      {emotion === 'sad' && (
        <mesh
          geometry={nodes.Cube045.geometry}
          material={materials.angry_mounth}
          position={[0.087, 1.26, 0.922]}
          rotation={[0, 0, 3.4]}
          scale={[0.2, 0.07, 0.1]}
        />
      )}

      {emotion === 'fear' && (
        <mesh
          geometry={nodes.Cube045.geometry}
          material={materials.angry_mounth}
          position={[0.087, 1.28, 0.922]}
          rotation={[0, 0, 3.0]}
          scale={[0.22, 0.11, 0.1]}
        />
      )}

      {emotion === 'surprise' && (
        <mesh
          geometry={nodes.Cube045.geometry}
          material={materials.angry_mounth}
          position={[0.087, 1.3, 0.922]}
          rotation={[0, 0, 3.14]}
          scale={[0.25, 0.15, 0.1]}
        />
      )}

      {/* อื่น ๆ */}
      <mesh
        geometry={nodes.Torus.geometry}
        material={materials['Material.017']}
        position={[0.016, 0.839, 0.015]}
        rotation={[0.316, 0.014, 0.015]}
        scale={0.386}
      />
    </group>
  )
}

useGLTF.preload('/FACE/Face.gltf')

export default Face;
