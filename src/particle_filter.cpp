/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <limits.h>
#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */


  num_particles = 50;  // TODO: Set the number of particles
  particles.resize(num_particles);
  weights.resize(num_particles);
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];

  std::default_random_engine gen;
  
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  for (int i = 0; i < num_particles; ++i)
  {
    Particle & particle = particles[i];
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;
    weights[i] = 1.0;
  }
  is_initialized = true;
}

void updatePosition(Particle& particle, double dt, double noise[], double velocity, double yaw_rate)
{
  // noise in the position of the particle.  
  if (yaw_rate == 0.0) {
    particle.x += velocity * dt * cos(particle.theta);
    particle.y += velocity * dt * sin(particle.theta);
  }
  else
  {
    particle.x += velocity / yaw_rate * (sin(particle.theta + yaw_rate * dt) - sin(particle.theta));
    particle.y += velocity / yaw_rate * (cos(particle.theta) - cos(particle.theta + dt * yaw_rate));
    particle.theta += dt * yaw_rate;
  }

  particle.x += noise[0];
  particle.y += noise[1];
  particle.theta += noise[2]; 
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];
  std::random_device device;
  std::default_random_engine gen(device());
  normal_distribution<double> dist_x(0.0, std_x);
  normal_distribution<double> dist_y(0.0, std_y);
  normal_distribution<double> dist_theta(0.0, std_theta);

  for (int i = 0; i < particles.size(); ++i)
  {
    // calculate the positional noises.
    double noises[3] = {dist_x(gen), dist_y(gen), dist_theta(gen)};
    updatePosition(particles[i], delta_t, noises, velocity, yaw_rate);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  for (int i = 0; i < observations.size(); ++i)
  {
    double closest = std::numeric_limits<double>::max();
    for (int j = 0; j < predicted.size(); ++j)
    {
      double temp_distance = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
      if (closest > temp_distance)
      {
        closest = temp_distance;
        observations[i].id = j;
      }
    }
  }

}

double bivariate_dist(double mean_x, double std_x, double mean_y, double std_y, double x, double y)
{
  return 1 / (2 * M_PI * std_x * std_y) * exp( -pow(x - mean_x, 2) / (2 * pow(std_x,2))  - pow(y - mean_y, 2)/ (2 * pow(std_y, 2)));
}

LandmarkObs transform_into_map_system(Particle const& particle, LandmarkObs const& landmarkObs)
{
  LandmarkObs transformed;
  transformed.id = landmarkObs.id;
  transformed.x = particle.x + cos(particle.theta) * landmarkObs.x - sin(particle.theta) * landmarkObs.y;
  transformed.y = particle.y + sin(particle.theta) * landmarkObs.x + cos(particle.theta) * landmarkObs.y;
  return transformed;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  
  /**
  * PLAN: from the map_landmarks and each particle position,
  * calculate the PREDICTED measurements. (remember it is restricted by the sensor range). (this is still in the particle reference) 
  * Then bring this to map coordinates
  *  Finally compare to observation measurements.
  */
  for (int p = 0; p < particles.size(); ++p)
  {
    Particle & particle = particles[p];
    vector<LandmarkObs> predicted;
    // store for landmarks in range of the particle sensor only
    for (int l = 0; l < map_landmarks.landmark_list.size(); ++l)
    {
      if (dist(particle.x, particle.y, map_landmarks.landmark_list[l].x_f, map_landmarks.landmark_list[l].y_f) <= sensor_range)
      {
        LandmarkObs predicted_landmark;
        predicted_landmark.id = map_landmarks.landmark_list[l].id_i;
        predicted_landmark.x = map_landmarks.landmark_list[l].x_f;
        predicted_landmark.y = map_landmarks.landmark_list[l].y_f;
               
        predicted.push_back(predicted_landmark);
      }
    }
    vector<LandmarkObs> map_observations;
    for (int l = 0; l < observations.size(); ++l)
    {
      map_observations.push_back(transform_into_map_system(particle, observations[l]));
    }

    dataAssociation(predicted, map_observations);

    weights[p] = 1.0;
    for (int l = 0; l < map_observations.size(); ++l)
    {
      LandmarkObs & cur_obs = map_observations[l];
      LandmarkObs & nearestPredicted = predicted[cur_obs.id];
      weights[p] *= bivariate_dist(cur_obs.x, std_landmark[0], cur_obs.y, std_landmark[1], nearestPredicted.x, nearestPredicted.y);
    }

    particle.weight = weights[p];
  }
  
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::random_device rd;
  std::default_random_engine generator(rd());
  std::discrete_distribution<int> distribution(weights.begin(), weights.end());
  std::vector<Particle> new_particles;
  for (int _ = 0; _ < num_particles; ++_)
  {
    int random_value = distribution(generator);
    new_particles.push_back(particles[random_value]);
  }
  particles = new_particles;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}