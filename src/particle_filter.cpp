/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

using namespace std;

LandmarkObs _rotate_translate(double x, double y, double theta, LandmarkObs observation) {

  double x_t = observation.x * cos(theta) - observation.y * sin(theta) + x;
  double y_t = observation.x * sin(theta) + observation.y * cos(theta) + y;

  LandmarkObs l;
  l.x = x_t;
  l.y = y_t;

  return l;
}

vector<LandmarkObs> _dataAssociation(const vector<LandmarkObs>& predicted, const vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.

  // obervations in the map coordiante system that are cloest to a given landmark
  vector<LandmarkObs> nearest_neighbors;

  for (auto &prediction: predicted) {
    unsigned int best_index = 0;
    double best_distance = numeric_limits<double>::max();

    for (unsigned int j = 0; j < observations.size(); ++j) {
      LandmarkObs observation = observations[j];
      double distance = dist(prediction.x, prediction.y, observation.x, observation.y);
      if (distance < best_distance) {
        best_distance = distance;
        best_index = j;
      }
    }

    nearest_neighbors.push_back(observations[best_index]);
  }
  return nearest_neighbors;
}

double _Gaussian(LandmarkObs measurement, LandmarkObs predicted_measurement, double std_landmark[]) {
  double std_x = std_landmark[0];
  double std_y = std_landmark[1];

  double delta_x = measurement.x - predicted_measurement.x;
  double delta_y = measurement.y - predicted_measurement.y;

  double x_y = ((delta_x * delta_x)/(std_x * std_x)) + ((delta_y * delta_y)/(std_y * std_y));

  double w = exp(-0.5 * x_y) / (2*M_PI*std_x*std_y);

  if (w < 0.0001) {
    w = 0.0001;
  }
  return w;
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  normal_distribution<double> N_x_init(x, std[0]);
  normal_distribution<double> N_y_init(y, std[1]);
  normal_distribution<double> N_theta_init(theta, std[2]);
  default_random_engine gen;

  num_particles = 100;
  for (int i = 0; i < num_particles; ++i) {
    Particle p {
      i,
      N_x_init(gen),
      N_y_init(gen),
      N_theta_init(gen),
      1
    };
    particles.push_back(p);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  normal_distribution<double> N_x_init(0, std_pos[0]);
  normal_distribution<double> N_y_init(0, std_pos[1]);
  normal_distribution<double> N_theta_init(0, std_pos[2]);
  default_random_engine gen;

  for (auto &p: particles) {
    double theta_new;
    double x_new;
    double y_new;
    if (fabs(yaw_rate) < 0.0001) {
      theta_new = p.theta;
      x_new = p.x + velocity * delta_t * cos(p.theta);
      y_new = p.y + velocity * delta_t * sin(p.theta);
    } else {
      theta_new = p.theta + yaw_rate * delta_t;
      x_new = p.x + (velocity / yaw_rate) * (sin(theta_new) - sin(p.theta));
      y_new = p.y + (velocity / yaw_rate) * (cos(p.theta) - cos(theta_new));
    }
    p.theta = theta_new + N_theta_init(gen);
    p.x = x_new + N_x_init(gen);
    p.y = y_new + N_x_init(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, vector<LandmarkObs>& observations) {}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

  for (unsigned long i = 0; i < particles.size(); ++i) {
    Particle particle = particles[i];

    vector<LandmarkObs> predicted_landmarks;
    for (auto &landmark: map_landmarks.landmark_list) {
      if (dist(particle.x, particle.y, landmark.x_f, landmark.y_f) < sensor_range) {
        LandmarkObs l;
        l.x = landmark.x_f;
        l.y = landmark.y_f;
        predicted_landmarks.push_back(l);
      }
    }

    vector<LandmarkObs> translated_observations;
    for (auto &observation: observations) {
      LandmarkObs l = _rotate_translate(particle.x, particle.y, particle.theta, observation);
      translated_observations.push_back(l);
    }

    // filtered and reordered translated_observations list where each index is the measurement
    // closest to the corresponding index in predicted_landmarks
    vector<LandmarkObs> closest_observations = _dataAssociation(predicted_landmarks, translated_observations);

    double weight = 1.0;

    for (unsigned long j = 0; j < closest_observations.size(); ++j) {
      LandmarkObs observation = closest_observations[j];
      LandmarkObs landmark = predicted_landmarks[j];

      weight *= _Gaussian(landmark, observation, std_landmark);
    }

    particle.weight = weight;
    weights[i] = weight;
  }

  // Normalize
  double weights_sum = accumulate(weights.begin(), weights.end(), 0.0);
  for(size_t i = 0; i < weights.size(); ++i) {
    particles[i].weight = particles[i].weight / weights_sum;
    weights[i] = weights[i] / weights_sum;
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  discrete_distribution<int> index_generator(weights.begin(), weights.end());
  random_device rd;
  mt19937 rng(rd());

  vector<Particle> new_particles;
  for (unsigned int i = 0; i < num_particles; ++i) {
    int next_index = index_generator(rng);
    Particle p = particles[next_index];
    Particle new_particle = Particle {
      p.id, p.x, p.y, p.theta, p.weight
    };
    new_particles.push_back(new_particle);
  }

  particles = new_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
