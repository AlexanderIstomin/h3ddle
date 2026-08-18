import Foundation

/// Offline EBU R128 integrated loudness for export normalization.
enum Loudness {
  static let targetLUFS = -14.0

  static func integratedLUFS(samples: [Float], sampleRate: Double, channels: Int) -> Double {
    guard sampleRate > 0 else { return -70 }
    var meter = Meter(sampleRate: sampleRate, channels: channels)
    meter.append(samples)
    return meter.integratedLUFS
  }

  static func gain(from measuredLUFS: Double, targetLUFS: Double = targetLUFS) -> Float {
    guard measuredLUFS.isFinite, measuredLUFS > -70 else { return 1 }
    let delta = min(20, max(-20, targetLUFS - measuredLUFS))
    return Float(pow(10, delta / 20))
  }

  private static func energyToLUFS(_ meanSquare: Double) -> Double {
    guard meanSquare > 1e-12 else { return -70 }
    return -0.691 + 10 * log10(meanSquare)
  }

  /// Incremental EBU R128 meter with memory bounded independently of program length.
  struct Meter {
    private static let minimumLUFS = -70.0
    private static let maximumLUFS = 70.0
    private static let bucketWidth = 0.01
    private static let bucketCount =
      Int((maximumLUFS - minimumLUFS) / bucketWidth) + 1

    private let channels: Int
    private let blockSize: Int
    private let hop: Int
    private var pre: [Biquad]
    private var highPass: [Biquad]
    private var window: [Double]
    private var windowIndex = 0
    private var windowCount = 0
    private var windowEnergy = 0.0
    private var framesUntilNextBlock = 0
    private var totalEnergy = 0.0
    private var totalSampleCount = 0
    private var buckets: [LoudnessBucket]

    init(sampleRate: Double, channels: Int) {
      let validRate = max(1, sampleRate)
      let validChannels = max(1, channels)
      let blockSize = max(1, Int((0.4 * validRate).rounded()))
      self.channels = validChannels
      self.blockSize = blockSize
      hop = max(1, blockSize / 4)
      pre = Array(repeating: Biquad.preFilter(sampleRate: validRate), count: validChannels)
      highPass = Array(
        repeating: Biquad.highPass(sampleRate: validRate),
        count: validChannels
      )
      window = Array(repeating: 0, count: blockSize)
      buckets = Array(repeating: LoudnessBucket(), count: Self.bucketCount)
    }

    mutating func append(_ samples: [Float]) {
      samples.withUnsafeBufferPointer { append($0) }
    }

    mutating func append(_ samples: UnsafeBufferPointer<Float>) {
      let frameCount = samples.count / channels
      for frame in 0..<frameCount {
        var frameEnergy = 0.0
        for channel in 0..<channels {
          let index = frame * channels + channel
          let staged = pre[channel].process(samples[index])
          let filtered = highPass[channel].process(staged)
          frameEnergy += Double(filtered) * Double(filtered)
        }
        appendFrame(energy: frameEnergy)
      }
    }

    var integratedLUFS: Double {
      guard totalSampleCount > 0 else { return -70 }
      guard windowCount >= blockSize else {
        return Loudness.energyToLUFS(totalEnergy / Double(totalSampleCount))
      }

      var absoluteEnergy = 0.0
      var absoluteCount = 0
      for bucket in buckets where bucket.count > 0 {
        absoluteEnergy += bucket.energy
        absoluteCount += bucket.count
      }
      guard absoluteCount > 0 else { return -70 }

      let absoluteMean = absoluteEnergy / Double(absoluteCount)
      let relativeThreshold = Loudness.energyToLUFS(absoluteMean) - 10
      var gatedEnergy = 0.0
      var gatedCount = 0
      for bucket in buckets where bucket.count > 0 {
        let bucketMean = bucket.energy / Double(bucket.count)
        if Loudness.energyToLUFS(bucketMean) > relativeThreshold {
          gatedEnergy += bucket.energy
          gatedCount += bucket.count
        }
      }
      guard gatedCount > 0 else { return Loudness.energyToLUFS(absoluteMean) }
      return Loudness.energyToLUFS(gatedEnergy / Double(gatedCount))
    }

    private mutating func appendFrame(energy: Double) {
      totalEnergy += energy
      totalSampleCount += channels

      if windowCount < blockSize {
        window[windowIndex] = energy
        windowEnergy += energy
        windowIndex = (windowIndex + 1) % blockSize
        windowCount += 1
        if windowCount == blockSize {
          recordCurrentBlock()
          framesUntilNextBlock = hop
        }
        return
      }

      windowEnergy -= window[windowIndex]
      window[windowIndex] = energy
      windowEnergy += energy
      windowIndex = (windowIndex + 1) % blockSize
      framesUntilNextBlock -= 1
      if framesUntilNextBlock == 0 {
        recordCurrentBlock()
        framesUntilNextBlock = hop
      }
    }

    private mutating func recordCurrentBlock() {
      let energy = windowEnergy / Double(blockSize * channels)
      let lufs = Loudness.energyToLUFS(energy)
      guard lufs > Self.minimumLUFS else { return }
      let clamped = min(max(lufs, Self.minimumLUFS), Self.maximumLUFS)
      let index = min(
        Self.bucketCount - 1,
        Int(((clamped - Self.minimumLUFS) / Self.bucketWidth).rounded(.down))
      )
      buckets[index].energy += energy
      buckets[index].count += 1
    }
  }
}

private struct LoudnessBucket {
  var energy = 0.0
  var count = 0
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
