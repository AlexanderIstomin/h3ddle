import Foundation

/// Offline EBU R128 integrated loudness for export normalization.
enum Loudness {
  static let targetLUFS = -14.0

  static func integratedLUFS(samples: [Float], sampleRate: Double, channels: Int) -> Double {
    let channels = max(1, channels)
    let frameCount = samples.count / channels
    guard frameCount > 0, sampleRate > 0 else { return -70 }

    var filtered = [Float](repeating: 0, count: samples.count)
    var pre = Array(repeating: Biquad.preFilter(sampleRate: sampleRate), count: channels)
    var highPass = Array(repeating: Biquad.highPass(sampleRate: sampleRate), count: channels)
    for frame in 0..<frameCount {
      for channel in 0..<channels {
        let index = frame * channels + channel
        let staged = pre[channel].process(samples[index])
        filtered[index] = highPass[channel].process(staged)
      }
    }

    let blockSize = max(1, Int((0.4 * sampleRate).rounded()))
    let hop = max(1, blockSize / 4)
    guard frameCount >= blockSize else {
      return ungatedLUFS(filtered, channels: channels)
    }

    var blockEnergies: [Double] = []
    var offset = 0
    while offset + blockSize <= frameCount {
      var energy = 0.0
      for frame in offset..<(offset + blockSize) {
        for channel in 0..<channels {
          let sample = Double(filtered[frame * channels + channel])
          energy += sample * sample
        }
      }
      blockEnergies.append(energy / Double(blockSize * channels))
      offset += hop
    }

    let absolute = blockEnergies.filter { energyToLUFS($0) > -70 }
    guard !absolute.isEmpty else { return -70 }
    let absoluteMean = absolute.reduce(0, +) / Double(absolute.count)
    let relativeThreshold = energyToLUFS(absoluteMean) - 10
    let gated = absolute.filter { energyToLUFS($0) > relativeThreshold }
    let mean = (gated.isEmpty ? absolute : gated).reduce(0, +) / Double((gated.isEmpty ? absolute : gated).count)
    return energyToLUFS(mean)
  }

  static func gain(from measuredLUFS: Double, targetLUFS: Double = targetLUFS) -> Float {
    guard measuredLUFS.isFinite, measuredLUFS > -70 else { return 1 }
    let delta = min(20, max(-20, targetLUFS - measuredLUFS))
    return Float(pow(10, delta / 20))
  }

  private static func ungatedLUFS(_ samples: [Float], channels: Int) -> Double {
    var energy = 0.0
    for sample in samples {
      energy += Double(sample) * Double(sample)
    }
    return energyToLUFS(energy / Double(max(1, samples.count)))
  }

  private static func energyToLUFS(_ meanSquare: Double) -> Double {
    guard meanSquare > 1e-12 else { return -70 }
    return -0.691 + 10 * log10(meanSquare)
  }
}

private struct Biquad {
  var b0: Float
  var b1: Float
  var b2: Float
  var a1: Float
  var a2: Float
  var z1: Float = 0
  var z2: Float = 0

  mutating func process(_ input: Float) -> Float {
    let output = b0 * input + z1
    z1 = b1 * input - a1 * output + z2
    z2 = b2 * input - a2 * output
    return output
  }

  /// ITU-R BS.1770-4 stage 1 high shelf.
  static func preFilter(sampleRate: Double) -> Biquad {
    designed(
      sampleRate: sampleRate,
      f0: 1_681.974450955533,
      gainDB: 3.999843853973347,
      q: 0.7071752369554196,
      highShelf: true
    )
  }

  /// ITU-R BS.1770-4 stage 2 high-pass.
  static func highPass(sampleRate: Double) -> Biquad {
    designed(
      sampleRate: sampleRate,
      f0: 38.13547087602444,
      gainDB: 0,
      q: 0.5003270373238773,
      highShelf: false
    )
  }

  private static func designed(
    sampleRate: Double,
    f0: Double,
    gainDB: Double,
    q: Double,
    highShelf: Bool
  ) -> Biquad {
    let k = tan(.pi * f0 / sampleRate)
    if highShelf {
      let v = pow(10.0, abs(gainDB) / 20.0)
      let sqrt2v = sqrt(2.0 * v)
      let denom = 1 + sqrt(2.0) * k + k * k
      let b0 = (v + sqrt2v * k + k * k) / denom
      let b1 = 2 * (k * k - v) / denom
      let b2 = (v - sqrt2v * k + k * k) / denom
      let a1 = 2 * (k * k - 1) / denom
      let a2 = (1 - sqrt(2.0) * k + k * k) / denom
      return Biquad(b0: Float(b0), b1: Float(b1), b2: Float(b2), a1: Float(a1), a2: Float(a2))
    }
    let denom = 1 + k / q + k * k
    let b0 = 1 / denom
    let b1 = -2 / denom
    let b2 = 1 / denom
    let a1 = 2 * (k * k - 1) / denom
    let a2 = (1 - k / q + k * k) / denom
    return Biquad(b0: Float(b0), b1: Float(b1), b2: Float(b2), a1: Float(a1), a2: Float(a2))
  }
}